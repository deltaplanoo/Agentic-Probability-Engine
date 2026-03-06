import os
import asyncio
import json
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from fastmcp import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

load_dotenv()

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    original_question: str
    search_query: str
    search_results: str
    parameters: list[dict]

# ── Nodes ─────────────────────────────────────────────────────────────────────

def make_nodes(model, search_tool):

    # STEP 1: Reword the user question into an optimal search query
    def reword_query(state: AgentState) -> dict:
        system = SystemMessage(content=(
            "You are a search query optimizer. "
            "Given a decision-making question about a physical location, "
            "rewrite it as a concise, effective web search query to find "
            "data useful for evaluating that location for a business decision. "
            "Return ONLY the search query string, nothing else."
        ))
        response = model.invoke([system, HumanMessage(content=state["original_question"])])
        query = response.content.strip()
        print(f"\n[Step 1] Reworded query: {query}")
        return {"search_query": query}

    # STEP 2: Call the web search tool once with the reworded query
    async def run_search(state: AgentState) -> dict:
        result = await search_tool.coroutine(query=state["search_query"])
        print(f"\n[Step 2] Search results received ({len(result)} chars)")
        return {"search_results": result}

    # STEP 3: Model freely decides which parameters are relevant, extracts + scores them
    def extract_and_score_parameters(state: AgentState) -> dict:
        system = SystemMessage(content=(
            "You are a business location analyst. "
            "Given a decision-making question and search results about a specific location, "
            "your job is to:\n\n"
            "1. Identify ALL parameters that are relevant to making THIS specific decision "
            "For example, for a restaurant: foot traffic, competition, rent, tourism, parking... "
            "For a gym: demographics, nearby gyms, accessibility, residential density... "
            "Use your judgment.\n\n"
            "2. For each parameter you identify:\n"
            "   - Extract its value from the search results (use 'No data found' if missing)\n"
            "   - Score it from 1 (very unfavorable) to 10 (very favorable) for this decision\n"
            "   - Assign a sentiment: 'positive', 'neutral', or 'negative'\n"
            "   - Write one concise sentence of reasoning\n\n"
            "3. Sort them by score descending.\n\n"
            "Return a JSON array of objects with this exact shape:\n"
            '[\n'
            '  {\n'
            '    "parameter": "parameter name",\n'
            '    "value": "what you found",\n'
            '    "score": 1-10,\n'
            '    "sentiment": "positive|neutral|negative",\n'
            '    "reasoning": "one sentence"\n'
            '  }\n'
            ']\n'
            "Return ONLY valid JSON. No markdown, no explanation."
        ))
        human = HumanMessage(content=(
            f"Decision question: {state['original_question']}\n\n"
            f"Search results:\n{state['search_results']}"
        ))
        response = model.invoke([system, human])

        try:
            raw = response.content.strip().replace("```json", "").replace("```", "")
            parameters = json.loads(raw)
            parameters.sort(key=lambda p: -p.get("score", 5))
        except json.JSONDecodeError:
            parameters = [{
                "parameter": "parse_error",
                "value": response.content,
                "score": 5,
                "sentiment": "neutral",
                "reasoning": "Could not parse model output."
            }]

        print(f"\n[Step 3] Extracted and scored {len(parameters)} parameters")
        return {"parameters": parameters}

    # STEP 4: Present final results
    def present_results(state: AgentState) -> dict:
        params = state["parameters"]
        avg_score = sum(p.get("score", 5) for p in params) / len(params) if params else 0
        sentiment_emoji = {"positive": "✅", "neutral": "⚪", "negative": "❌"}

        print(f"\n{'='*65}")
        print(f" LOCATION ANALYSIS")
        print(f" {state['original_question']}")
        print(f"{'='*65}\n")

        for p in params:
            emoji = sentiment_emoji.get(p.get("sentiment", "neutral"), "⚪")
            score = p.get("score", "?")
            print(f"{emoji} [{score}/10] {p['parameter']}")
            print(f"         Value:    {p['value']}")
            print(f"         Reason:   {p.get('reasoning', '-')}\n")

        print(f"{'='*65}")
        print(f" OVERALL SCORE: {avg_score:.1f}/10")
        verdict = (
            "✅ Looks promising" if avg_score >= 7
            else "⚪  Mixed signals" if avg_score >= 5
            else "❌ High risk"
        )
        print(f" VERDICT: {verdict}")
        print(f"{'='*65}\n")
        return {}

    return reword_query, run_search, extract_and_score_parameters, present_results


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_agent(question: str):
    async with Client("http://localhost:8000/sse") as mcp_client:

        raw_mcp_tools = await mcp_client.list_tools()
        print(f"Found {len(raw_mcp_tools)} MCP tools: {[t.name for t in raw_mcp_tools]}")

        class WebSearchInput(BaseModel):
            query: str = Field(description="The search query to look up")

        def make_wrapper(name):
            async def wrapper(query: str) -> str:
                result = await mcp_client.call_tool(name, arguments={"query": query})
                return result.content[0].text
            return wrapper

        search_tool = StructuredTool.from_function(
            name="web_search",
            description="Search the web for information",
            coroutine=make_wrapper("web_search"),
            args_schema=WebSearchInput,
        )

        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        reword_query, run_search, extract_and_score_parameters, present_results = make_nodes(model, search_tool)

        workflow = StateGraph(AgentState)
        workflow.add_node("reword_query", reword_query)
        workflow.add_node("run_search", run_search)
        workflow.add_node("extract_and_score_parameters", extract_and_score_parameters)
        workflow.add_node("present_results", present_results)

        workflow.add_edge(START, "reword_query")
        workflow.add_edge("reword_query", "run_search")
        workflow.add_edge("run_search", "extract_and_score_parameters")
        workflow.add_edge("extract_and_score_parameters", "present_results")
        workflow.add_edge("present_results", END)

        app = workflow.compile()

        await app.ainvoke({
            "messages": [HumanMessage(content=question)],
            "original_question": question,
            "search_query": "",
            "search_results": "",
            "parameters": [],
        })


if __name__ == "__main__":
    asyncio.run(run_agent(
        "Is opening a restaurant in Via Guido Reni 2 in Pontassieve a good idea?"
    ))