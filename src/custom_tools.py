import asyncio
import json
import logging
import os

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

SEARCH_SYSTEM_PROMPT = """You are a research analyst gathering evidence to support a business or \
location decision (e.g. opening a shop, renting a property, entering a market). Use Google Search \
to gather current, relevant evidence about the user's query.

Guidelines:
- Prioritize recent and authoritative sources: official institutions, public authorities, chambers \
of commerce, municipalities, universities, industry associations, statistical offices, direct \
company sources, and established publications.
- If the query concerns an address, neighborhood, city, or province, prioritize geographically \
precise information over generic national/international data.
- Include dates and reference years for every data point you cite.
- Clearly distinguish verified facts, estimates, opinions, and information that is missing or \
uncertain.
- If sources disagree, report the conflicting evidence rather than picking a side.
- Do not give a final decision verdict — you are gathering evidence, not deciding.
- Avoid generic SEO/marketing content when stronger, more authoritative sources are available.
- Collect useful quantitative data wherever available (numbers, percentages, prices, counts, dates).

Structure your answer as:
1. Main evidence
2. Quantitative data
3. Favorable elements
4. Critical elements
5. Missing or uncertain information
"""

_TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "unavailable",
    "deadline exceeded",
    "resource exhausted",
)

_NON_RETRYABLE_ERROR_MARKERS = (
    "api key",
    "401",
    "403",
    "permission",
    "invalid argument",
    "400",
    "404",
    "unsupported model",
)


def build_default_search_model() -> ChatGoogleGenerativeAI:
    """Build the default Gemini model used for grounded web search.

    Relies on ChatGoogleGenerativeAI's built-in GOOGLE_API_KEY -> GEMINI_API_KEY
    env var auto-detection instead of reading/passing the key ourselves.
    max_retries=1 disables the SDK's own internal retry so our retry budget
    in _ainvoke_with_retry is the only one in play.
    """
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_SEARCH_MODEL", "gemini-2.5-flash"),
        temperature=0.1,
        max_retries=1,
    )


def extract_answer_text(response) -> str:
    """Best-effort extraction of the grounded answer text. Never raises."""
    text_accessor = getattr(response, "text", None)
    if callable(text_accessor):
        try:
            text_accessor = text_accessor()
        except TypeError:
            pass
    if text_accessor:
        text = str(text_accessor).strip()
        if text:
            return text

    try:
        blocks = response.content_blocks or []
    except Exception:
        blocks = []
    parts = [
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    joined = "".join(parts).strip()
    if joined:
        return joined

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts).strip()

    return ""


def extract_google_grounding(response) -> tuple[list[dict], list[str]]:
    """Extract deduplicated sources and executed Google Search queries.

    Primary source of truth is response.response_metadata["grounding_metadata"]
    (grounding_chunks / web_search_queries), which is populated whenever grounding
    occurred. response.content_blocks citation annotations are only used to
    enrich cited_text — they can be empty even when real sources exist (Gemini
    only emits them when it correlates a text span to a source).
    """
    sources: list[dict] = []
    seen_urls: set[str] = set()

    try:
        grounding_metadata = getattr(response, "response_metadata", {}).get("grounding_metadata") or {}
    except Exception:
        grounding_metadata = {}

    for chunk in grounding_metadata.get("grounding_chunks") or []:
        web_info = (chunk or {}).get("web") or {}
        url = web_info.get("uri")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        sources.append({"title": web_info.get("title") or "", "url": url, "cited_text": ""})

    try:
        blocks = response.content_blocks or []
    except Exception:
        blocks = []

    cited_text_by_url: dict[str, str] = {}
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        for annotation in block.get("annotations") or []:
            if not isinstance(annotation, dict) or annotation.get("type") != "citation":
                continue
            url = annotation.get("url")
            cited_text = annotation.get("cited_text")
            if not url:
                continue
            if cited_text and url not in cited_text_by_url:
                cited_text_by_url[url] = cited_text
            if url not in seen_urls:
                seen_urls.add(url)
                sources.append(
                    {"title": annotation.get("title") or "", "url": url, "cited_text": cited_text or ""}
                )

    for source in sources:
        if not source["cited_text"] and source["url"] in cited_text_by_url:
            source["cited_text"] = cited_text_by_url[source["url"]]

    queries: list[str] = []
    seen_queries: set[str] = set()
    for query in grounding_metadata.get("web_search_queries") or []:
        if query and query not in seen_queries:
            seen_queries.add(query)
            queries.append(query)

    return sources, queries


def _is_transient_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if any(marker in message for marker in _NON_RETRYABLE_ERROR_MARKERS):
        return False
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


async def _ainvoke_with_retry(model, messages, max_attempts: int = 2):
    """Invoke the model with a single shared retry budget covering both
    transient exceptions and empty-answer responses."""
    response = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = await model.ainvoke(messages)
        except Exception as exc:
            if attempt >= max_attempts or not _is_transient_error(exc):
                raise
            logger.warning("Transient error on attempt %d/%d: %s", attempt, max_attempts, exc)
            continue

        if extract_answer_text(response):
            return response

        if attempt >= max_attempts:
            return response
        logger.warning("Empty answer on attempt %d/%d, retrying", attempt, max_attempts)

    return response


def _error_payload(query: str, model_name: str, message: str) -> str:
    return json.dumps(
        {
            "provider": "gemini_google_search",
            "model": model_name,
            "query": query,
            "error": message,
        },
        ensure_ascii=False,
        indent=2,
    )


async def perform_web_search(query: str, grounded_search_model, semaphore: asyncio.Semaphore, model_name: str) -> str:
    """Core web_search logic, standalone for testability independent of FastMCP."""
    if not query or not query.strip():
        return _error_payload(query, model_name, "Query must not be empty.")

    messages = [SystemMessage(content=SEARCH_SYSTEM_PROMPT), HumanMessage(content=query)]

    try:
        async with semaphore:
            response = await _ainvoke_with_retry(grounded_search_model, messages)
    except Exception as exc:
        logger.warning("web_search failed for query=%r: %s: %s", query, type(exc).__name__, exc)
        return _error_payload(query, model_name, f"{type(exc).__name__}: {exc}")

    answer = extract_answer_text(response)
    if not answer:
        logger.warning("web_search returned no usable text for query=%r", query)
        return _error_payload(query, model_name, "Model returned no usable text.")

    sources, search_queries = extract_google_grounding(response)

    payload = {
        "provider": "gemini_google_search",
        "model": model_name,
        "query": query,
        "answer": answer,
        "search_queries": search_queries,
        "sources": sources,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def register_tools(mcp, search_model=None):
    """
    Registra i tool di ricerca, albero decisionale e geocodifica
    sull'istanza MCP passata come argomento.
    """
    if search_model is None:
        search_model = build_default_search_model()

    grounded_search_model = search_model.bind_tools([{"google_search": {}}])
    model_name = getattr(search_model, "model", None) or os.getenv("GEMINI_SEARCH_MODEL", "gemini-2.5-flash")
    semaphore = asyncio.Semaphore(int(os.getenv("WEB_SEARCH_MAX_CONCURRENCY", "4")))

    @mcp.tool()
    async def web_search(query: str) -> str:
        """
        Performs a Gemini-grounded Google Search and returns a JSON payload
        with a decision-analysis answer, executed search queries, and sources.
        """
        return await perform_web_search(query, grounded_search_model, semaphore, model_name)

    @mcp.tool()
    def process_decision_tree(tree_structure: str) -> str:
        """
        Calculates Italian Flag triplets (favor, neutral, unfavor) for every
        non-leaf node by averaging children triplets, weighted only by node weight.
        IF values themselves are plain probabilities — no internal weighting.
        """
        def calculate_node(node):
            children = node.get("children", [])

            if not children:
                return (
                    node.get("favor",   0.0),
                    node.get("neutral", 0.0),
                    node.get("unfavor", 0.0),
                )

            total_weight = sum(abs(child.get("weight", 0.0)) for child in children)

            favor   = 0.0
            neutral = 0.0
            unfavor = 0.0

            for child in children:
                f, n, u = calculate_node(child)
                w = abs(child.get("weight", 0.0)) / total_weight if total_weight > 0 else 1 / len(children)
                favor   += f * w
                neutral += n * w
                unfavor += u * w

            node["favor"]   = round(favor,   4)
            node["neutral"] = round(neutral, 4)
            node["unfavor"] = round(unfavor, 4)
            return favor, neutral, unfavor

        try:
            data = json.loads(tree_structure)
            calculate_node(data)
            return json.dumps(data, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Invalid tree structure: {str(e)}"})

    @mcp.tool()
    async def geocode_nominatim(address: str, city: str = "", province: str = "") -> dict:
        """
        Geocodes an address using the Nominatim OpenStreetMap API.
        Returns GeoJSON-style result.
        """
        query_parts = [address]
        if city: query_parts.append(city)
        if province: query_parts.append(province)
        full_query = ", ".join(query_parts)

        params = {
            "q": full_query,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": 1
        }

        headers = {
            "User-Agent": "AgenticProbabilityEngine",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                return {"error": "No results found"}

            result = data[0]
            return {
                "lat": float(result["lat"]),
                "lon": float(result["lon"]),
                "display_name": result.get("display_name"),
                "raw": result
            }
