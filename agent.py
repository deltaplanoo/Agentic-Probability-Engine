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
from db import init_db, save_template, load_template, print_templates

load_dotenv()

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:          Annotated[list[BaseMessage], add_messages]
    original_question: str
    decision_type:     str        # e.g. "open a restaurant"
    variables:         dict       # e.g. {"address": "Via Roma 100 in Scandicci"}
    search_query:      str
    search_results:    str
    parameters:        list[dict]
    candidate_trees:   list[dict]
    decision_tree:     dict
    tree_reused:       bool

# ── Helpers ───────────────────────────────────────────────────────────────────

def inject_variables(node: dict, variables: dict) -> dict:
    """Recursively replace {variable} placeholders in all label and search_hint fields."""
    node = dict(node)
    for key in ("label", "search_hint"):
        if key in node and isinstance(node[key], str):
            for var_name, var_value in variables.items():
                node[key] = node[key].replace(f"{{{var_name}}}", var_value)
    node["children"] = [inject_variables(c, variables) for c in node.get("children", [])]
    return node

def collect_leaves(node: dict) -> list[dict]:
    if not node.get("children"):
        return [node]
    leaves = []
    for child in node.get("children", []):
        leaves.extend(collect_leaves(child))
    return leaves

def update_leaf_in_tree(node: dict, leaf_id: str, favor: float, neutral: float, unfavor: float) -> dict:
    """Return a new tree with IF values updated on the matching leaf."""
    node = dict(node)
    if node.get("id") == leaf_id:
        node["favor"]   = favor
        node["neutral"] = neutral
        node["unfavor"] = unfavor
        return node
    node["children"] = [
        update_leaf_in_tree(c, leaf_id, favor, neutral, unfavor)
        for c in node.get("children", [])
    ]
    return node

# ── Nodes ─────────────────────────────────────────────────────────────────────

def make_nodes(model, mcp_client, search_tool):

    # STEP 1: Parse question — extract decision_type + variables, check DB
    def parse_question(state: AgentState) -> dict:
        system = SystemMessage(content=(
            "You are a decision analysis expert.\n\n"
            "Given a decision-making question, extract:\n"
            "1. 'decision_type': the generic decision being made, stripped of all specific "
            "values. Lowercase short phrase.\n"
            "   Examples:\n"
            "   - 'Is opening a restaurant in Via Calzaiuoli 50 Florence a good idea?' "
            "→ 'open a restaurant'\n"
            "   - 'Should I open a gym in Berlin Mitte?' → 'open a gym'\n\n"
            "2. 'variables': a dict of named variable values extracted from the question.\n"
            "   Always include 'address' if a location is mentioned.\n"
            "   Examples:\n"
            "   - address: 'Via Calzaiuoli 50, Florence'\n"
            "   - address: 'Berlin Mitte'\n\n"
            "Return ONLY valid JSON in this shape:\n"
            '{"decision_type": "...", "variables": {"address": "..."}}\n'
            "No markdown, no explanation."
        ))
        response = model.invoke([system, HumanMessage(content=state["original_question"])])

        try:
            raw = response.content.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(raw)
            decision_type = parsed.get("decision_type", "").lower().strip()
            variables     = parsed.get("variables", {})
        except (json.JSONDecodeError, AttributeError):
            decision_type = "unknown decision"
            variables     = {}

        print(f"\n[Step 1] Decision type: '{decision_type}'")
        print(f"[Step 1] Variables: {variables}")

        # Check DB for existing template
        template = load_template(decision_type)
        if template:
            print(f"\n[Step 1] ✅ Template found in DB — injecting variables")
            injected_tree = inject_variables(template["tree"], variables)
            return {
                "decision_type": decision_type,
                "variables":     variables,
                "decision_tree": injected_tree,
                "tree_reused":   True,
            }
        else:
            print(f"\n[Step 1] No template found — will generate new one")
            return {
                "decision_type": decision_type,
                "variables":     variables,
                "decision_tree": {},
                "tree_reused":   False,
            }

    # STEP 2: Reword question into search query (first-run only)
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
        print(f"\n[Step 2] Reworded query: {query}")
        return {"search_query": query}

    # STEP 3: Global web search (first-run only)
    async def run_search(state: AgentState) -> dict:
        result = await search_tool.coroutine(query=state["search_query"])
        print(f"\n[Step 3] Search results received ({len(result)} chars)")
        return {"search_results": result}

    # STEP 4: Extract parameters with IF triplets (first-run only)
    def extract_and_score_parameters(state: AgentState) -> dict:
        system = SystemMessage(content=(
            "You are a business location analyst using the Italian Flag (IF) method.\n\n"
            "Given a decision-making question and search results about a specific location:\n\n"
            "1. Identify ALL parameters relevant to THIS specific decision.\n\n"
            "2. For each parameter assign an Italian Flag triplet:\n"
            "   - 'favor': probability this parameter supports the decision (0.0-1.0)\n"
            "   - 'neutral': probability this is uncertain or irrelevant (0.0-1.0)\n"
            "   - 'unfavor': probability this works against the decision (0.0-1.0)\n"
            "   - favor + neutral + unfavor MUST sum to exactly 1.0\n"
            "   - If data is missing, push into neutral\n\n"
            "3. Sort parameters by favor descending.\n\n"
            "Return a JSON array:\n"
            '[\n'
            '  {\n'
            '    "parameter": "parameter name",\n'
            '    "value": "what you found",\n'
            '    "favor": 0.0,\n'
            '    "neutral": 0.0,\n'
            '    "unfavor": 0.0,\n'
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
            for p in parameters:
                total = p.get("favor", 0.0) + p.get("neutral", 0.0) + p.get("unfavor", 0.0)
                if total > 0:
                    p["favor"]   = round(p["favor"]   / total, 4)
                    p["neutral"] = round(p["neutral"] / total, 4)
                    p["unfavor"] = round(p["unfavor"] / total, 4)
                else:
                    p["favor"], p["neutral"], p["unfavor"] = 0.0, 1.0, 0.0
            parameters.sort(key=lambda p: -p.get("favor", 0.0))
        except json.JSONDecodeError:
            parameters = [{
                "parameter": "parse_error",
                "value": response.content,
                "favor": 0.0, "neutral": 1.0, "unfavor": 0.0,
                "reasoning": "Could not parse model output."
            }]

        print(f"\n[Step 4] Extracted {len(parameters)} parameters")
        return {"parameters": parameters}

    # STEP 5a: Generate 3 candidate trees in parallel (first-run only)
    async def generate_decision_trees(state: AgentState) -> dict:
        params_json = json.dumps(state["parameters"], indent=2)

        tree_prompt_system = SystemMessage(content=(
            "You are a decision analysis expert.\n\n"
            "Build a weighted decision tree from the given parameters.\n\n"
            "Rules:\n"
            "- Root node = the final decision\n"
            "- Group related parameters under intermediate parent nodes\n"
            "- Leaf nodes = individual parameters\n"
            "- 'weight': importance among siblings (siblings must sum to 1.0)\n"
            "- Leave ALL favor/neutral/unfavor as 0.0 on every node\n"
            "- Do NOT add search_hint yet — leave it as empty string on leaves\n\n"
            "Use this exact JSON shape:\n"
            "{\n"
            '  "id": "root",\n'
            '  "label": "decision question",\n'
            '  "weight": 1.0,\n'
            '  "favor": 0.0, "neutral": 0.0, "unfavor": 0.0,\n'
            '  "children": [\n'
            "    {\n"
            '      "id": "group_1",\n'
            '      "label": "Group Name",\n'
            '      "weight": 0.6,\n'
            '      "favor": 0.0, "neutral": 0.0, "unfavor": 0.0,\n'
            '      "children": [\n'
            "        {\n"
            '          "id": "leaf_1",\n'
            '          "label": "Parameter Name",\n'
            '          "search_hint": "",\n'
            '          "weight": 0.5,\n'
            '          "favor": 0.0, "neutral": 0.0, "unfavor": 0.0,\n'
            '          "children": []\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Return ONLY valid JSON. No markdown, no explanation."
        ))
        human = HumanMessage(content=(
            f"Decision question: {state['original_question']}\n\n"
            f"Parameters to use as leaves:\n{params_json}"
        ))

        async def single_generation(i: int):
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, lambda: model.invoke([tree_prompt_system, human])
            )
            try:
                raw = response.content.strip().replace("```json", "").replace("```", "")
                tree = json.loads(raw)
                print(f"\n[Step 5a] Candidate {i+1} generated")
                return tree
            except json.JSONDecodeError:
                print(f"\n[Step 5a] Candidate {i+1} failed to parse")
                return None

        candidates = await asyncio.gather(*[single_generation(i) for i in range(3)])
        candidates = [c for c in candidates if c is not None]
        print(f"\n[Step 5a] {len(candidates)} valid candidates generated")
        return {"candidate_trees": candidates}

    # STEP 5b: Pick best candidate tree (first-run only)
    def pick_best_tree(state: AgentState) -> dict:
        candidates = state["candidate_trees"]
        if len(candidates) == 1:
            print(f"\n[Step 5b] Only 1 candidate, using directly")
            return {"decision_tree": candidates[0]}

        system = SystemMessage(content=(
            "You are a decision analysis expert.\n\n"
            "Pick the BEST decision tree from the candidates based on:\n"
            "1. Coverage — all relevant parameters included as leaves\n"
            "2. Structure — parameters grouped logically\n"
            "3. Weights — siblings sum to 1.0, sensibly distributed\n"
            "4. Depth — neither too flat nor too deep\n\n"
            "Return ONLY the index (0, 1, or 2). No explanation, just the integer."
        ))
        human = HumanMessage(content=(
            f"Decision question: {state['original_question']}\n\n"
            f"Candidates:\n{json.dumps(candidates, indent=2)}"
        ))
        response = model.invoke([system, human])

        try:
            best_index = int(response.content.strip())
            best_index = max(0, min(best_index, len(candidates) - 1))
        except ValueError:
            best_index = 0

        print(f"\n[Step 5b] Best tree: candidate {best_index + 1}")
        return {"decision_tree": candidates[best_index]}

    # STEP 6: Annotate leaves with parameterized search_hints, then save template
    def annotate_and_save_template(state: AgentState) -> dict:
        tree_json = json.dumps(state["decision_tree"], indent=2)
        variables = state["variables"]

        system = SystemMessage(content=(
            "You are a decision analysis expert.\n\n"
            "You will receive a decision tree and a variables dict.\n\n"
            "Your job:\n"
            "1. For every LEAF node, write a 'search_hint' — a web search query "
            "that would find data relevant to scoring that leaf.\n"
            "2. The search_hint MUST be parameterized: replace any specific variable "
            "values (like a specific address) with their placeholder (e.g. {address}).\n"
            "3. Also replace variable values in leaf and group LABELS with placeholders.\n"
            "4. Do NOT touch weights or IF values.\n"
            "5. Do NOT add search_hint to intermediate or root nodes.\n\n"
            "Examples of correct search_hints:\n"
            "  Leaf 'Foot Traffic': search_hint = 'pedestrian foot traffic {address}'\n"
            "  Leaf 'Competition': search_hint = 'restaurants near {address}'\n"
            "  Leaf 'Parking': search_hint = 'parking availability near {address}'\n"
            "  Leaf 'Rent': search_hint = 'commercial rent {address}'\n\n"
            "Return the complete updated tree as ONLY valid JSON. No markdown, no explanation."
        ))
        human = HumanMessage(content=(
            f"Variables: {json.dumps(variables)}\n\n"
            f"Tree:\n{tree_json}"
        ))
        response = model.invoke([system, human])

        try:
            raw = response.content.strip().replace("```json", "").replace("```", "")
            annotated_tree = json.loads(raw)
        except json.JSONDecodeError:
            print(f"\n[Step 6] Failed to parse annotated tree, using original")
            annotated_tree = state["decision_tree"]

        # Extract variable names from the variables dict
        variable_names = list(variables.keys())

        # Save parameterized template to DB
        save_template(state["decision_type"], variable_names, annotated_tree)
        print(f"\n[Step 6] Template annotated and saved for '{state['decision_type']}'")

        print_templates()

        # Re-inject variables so the current run uses concrete values
        injected_tree = inject_variables(annotated_tree, variables)
        return {"decision_tree": injected_tree}

    # STEP 7: Score IF on each leaf via individual targeted searches
    async def score_leaf_if(state: AgentState) -> dict:
        tree     = state["decision_tree"]
        leaves   = collect_leaves(tree)
        variables = state["variables"]

        print(f"\n[Step 7] Scoring {len(leaves)} leaves individually...")

        async def score_single_leaf(leaf: dict) -> tuple[str, float, float, float]:
            leaf_id    = leaf["id"]
            leaf_label = leaf["label"]
            hint       = leaf.get("search_hint", leaf_label)

            # Search
            try:
                search_result = await search_tool.coroutine(query=hint)
            except Exception:
                search_result = ""

            if not search_result.strip():
                print(f"  [{leaf_label}] No data — marking neutral")
                return (leaf_id, 0.0, 1.0, 0.0)

            # Score IF for this leaf
            system = SystemMessage(content=(
                "You are a decision analyst using the Italian Flag (IF) method.\n\n"
                "Given search results for a specific decision parameter, assign:\n"
                "- 'favor': probability this supports the decision (0.0-1.0)\n"
                "- 'neutral': probability this is uncertain/irrelevant (0.0-1.0)\n"
                "- 'unfavor': probability this works against the decision (0.0-1.0)\n"
                "- favor + neutral + unfavor MUST sum to exactly 1.0\n"
                "- If data is missing or unclear, push into neutral\n\n"
                "Return ONLY valid JSON: "
                '{"favor": 0.0, "neutral": 0.0, "unfavor": 0.0}\n'
                "No markdown, no explanation."
            ))
            human = HumanMessage(content=(
                f"Decision: {state['original_question']}\n"
                f"Parameter: {leaf_label}\n\n"
                f"Search results:\n{search_result}"
            ))
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, lambda: model.invoke([system, human])
            )

            try:
                raw = response.content.strip().replace("```json", "").replace("```", "")
                scored = json.loads(raw)
                f = scored.get("favor",   0.0)
                n = scored.get("neutral", 0.0)
                u = scored.get("unfavor", 0.0)
                total = f + n + u
                if total > 0:
                    f, n, u = round(f/total, 4), round(n/total, 4), round(u/total, 4)
                else:
                    f, n, u = 0.0, 1.0, 0.0
            except (json.JSONDecodeError, AttributeError):
                f, n, u = 0.0, 1.0, 0.0

            print(f"  [{leaf_label}] favor={f} neutral={n} unfavor={u}")
            return (leaf_id, f, n, u)

        # Run all leaf searches + scoring in parallel
        results = await asyncio.gather(*[score_single_leaf(leaf) for leaf in leaves])

        # Update tree with scored IF values
        updated_tree = tree
        for leaf_id, f, n, u in results:
            updated_tree = update_leaf_in_tree(updated_tree, leaf_id, f, n, u)

        print(f"\n[Step 7] All leaves scored")
        return {"decision_tree": updated_tree}

    # STEP 8: Propagate IF from leaves to root via MCP tool
    async def calculate_tree(state: AgentState) -> dict:
        tree_str = json.dumps(state["decision_tree"])
        result = await mcp_client.call_tool(
            "process_decision_tree",
            arguments={"tree_structure": tree_str}
        )
        calculated = json.loads(result.content[0].text)
        root = calculated
        print(f"\n[Step 8] IF propagated to root")
        print(f"         favor={root.get('favor','?')}  "
              f"neutral={root.get('neutral','?')}  "
              f"unfavor={root.get('unfavor','?')}")
        return {"decision_tree": calculated}

    # STEP 9: Present results
    def present_results(state: AgentState) -> dict:
        def if_bar(favor, neutral, unfavor, width=40):
            g = int(round(favor   * width))
            w = int(round(neutral * width))
            r = int(round(unfavor * width))
            total = g + w + r
            if total < width:
                w += width - total
            elif total > width:
                w -= total - width
            return f"\033[32m{'█'*g}\033[0m\033[37m{'█'*w}\033[0m\033[31m{'█'*r}\033[0m"

        def if_label(favor, unfavor):
            if favor > unfavor and favor > 0.5:
                return "✅ Favorable"
            elif unfavor > favor and unfavor > 0.5:
                return "❌ Unfavorable"
            elif favor > 0.4 and unfavor < 0.2:
                return "✅ Leaning favorable"
            elif unfavor > 0.4 and favor < 0.2:
                return "❌ Leaning unfavorable"
            else:
                return "⚪ Uncertain"

        tree   = state["decision_tree"]
        leaves = collect_leaves(tree)
        reused = state.get("tree_reused", False)

        print(f"\n{'='*65}")
        print(f" LOCATION ANALYSIS — Italian Flag Method")
        print(f" {state['original_question']}")
        print(f" {'[template reused]' if reused else '[new template generated]'}")
        print(f"{'='*65}\n")

        print(f"  {'PARAMETER':<26}  {'FAV':>5} {'NEU':>5} {'UNF':>5}  FLAG")
        print(f"  {'-'*61}")

        for leaf in leaves:
            f = leaf.get("favor",   0.0)
            n = leaf.get("neutral", 0.0)
            u = leaf.get("unfavor", 0.0)
            bar   = if_bar(f, n, u)
            label = if_label(f, u)
            print(f"  {leaf['label']:<26}  {f:>5.2f} {n:>5.2f} {u:>5.2f}  {bar}  {label}")

        rf = tree.get("favor",   0.0)
        rn = tree.get("neutral", 0.0)
        ru = tree.get("unfavor", 0.0)

        print(f"\n{'='*65}")
        print(f" DECISION TREE RESULT")
        print(f"{'='*65}")
        print(f"  Favor    {rf:.3f}  {if_bar(rf, 0,  0,  width=40)}")
        print(f"  Neutral  {rn:.3f}  {if_bar(0,  rn, 0,  width=40)}")
        print(f"  Unfavor  {ru:.3f}  {if_bar(0,  0,  ru, width=40)}")
        print(f"\n  Full flag:  {if_bar(rf, rn, ru, width=50)}")
        print(f"\n  VERDICT: {if_label(rf, ru)}")
        print(f"{'='*65}\n")
        return {}

    return (
        parse_question,
        reword_query,
        run_search,
        extract_and_score_parameters,
        generate_decision_trees,
        pick_best_tree,
        annotate_and_save_template,
        score_leaf_if,
        calculate_tree,
        present_results,
    )

# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_parse(state: AgentState) -> str:
    return "score_leaf_if" if state.get("tree_reused") else "reword_query"

# ── Main ──────────────────────────────────────────────────────────────────────

async def run_agent(question: str):
    init_db()

    mcp_client = Client("http://localhost:8000/sse")
    await mcp_client.__aenter__()

    try:
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

        (
            parse_question,
            reword_query,
            run_search,
            extract_and_score_parameters,
            generate_decision_trees,
            pick_best_tree,
            annotate_and_save_template,
            score_leaf_if,
            calculate_tree,
            present_results,
        ) = make_nodes(model, mcp_client, search_tool)

        workflow = StateGraph(AgentState)

        workflow.add_node("parse_question",               parse_question)
        workflow.add_node("reword_query",                 reword_query)
        workflow.add_node("run_search",                   run_search)
        workflow.add_node("extract_and_score_parameters", extract_and_score_parameters)
        workflow.add_node("generate_decision_trees",      generate_decision_trees)
        workflow.add_node("pick_best_tree",               pick_best_tree)
        workflow.add_node("annotate_and_save_template",   annotate_and_save_template)
        workflow.add_node("score_leaf_if",                score_leaf_if)
        workflow.add_node("calculate_tree",               calculate_tree)
        workflow.add_node("present_results",              present_results)

        workflow.add_edge(START, "parse_question")
        workflow.add_conditional_edges(
            "parse_question",
            route_after_parse,
            {
                "reword_query":  "reword_query",
                "score_leaf_if": "score_leaf_if",
            }
        )
        workflow.add_edge("reword_query",                 "run_search")
        workflow.add_edge("run_search",                   "extract_and_score_parameters")
        workflow.add_edge("extract_and_score_parameters", "generate_decision_trees")
        workflow.add_edge("generate_decision_trees",      "pick_best_tree")
        workflow.add_edge("pick_best_tree",               "annotate_and_save_template")
        workflow.add_edge("annotate_and_save_template",   "score_leaf_if")
        workflow.add_edge("score_leaf_if",                "calculate_tree")
        workflow.add_edge("calculate_tree",               "present_results")
        workflow.add_edge("present_results",              END)

        app = workflow.compile()

        await app.ainvoke({
            "messages":          [HumanMessage(content=question)],
            "original_question": question,
            "decision_type":     "",
            "variables":         {},
            "search_query":      "",
            "search_results":    "",
            "parameters":        [],
            "candidate_trees":   [],
            "decision_tree":     {},
            "tree_reused":       False,
        })

    finally:
        await mcp_client.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(run_agent(
        "Is opening a restaurant in Via Calzaiuoli 50 in Florence a good idea?"
    ))