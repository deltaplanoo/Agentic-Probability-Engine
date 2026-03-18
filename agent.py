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
    decision_tree: dict

# ── Nodes ─────────────────────────────────────────────────────────────────────

def make_nodes(model, mcp_client, search_tool):

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

    # STEP 3: Extract parameters with IF triplets
    def extract_and_score_parameters(state: AgentState) -> dict:
        system = SystemMessage(content=(
            "You are a business location analyst using the Italian Flag (IF) method.\n\n"
            "Given a decision-making question and search results about a specific location:\n\n"
            "1. Identify ALL parameters relevant to THIS specific decision "
            "(e.g. for a restaurant: foot traffic, competition, rent, tourism, parking...)\n\n"
            "2. For each parameter assign an Italian Flag triplet where:\n"
            "   - 'favor': probability this parameter supports the decision (0.0 to 1.0)\n"
            "   - 'neutral': probability this parameter is uncertain or irrelevant (0.0 to 1.0)\n"
            "   - 'unfavor': probability this parameter works against the decision (0.0 to 1.0)\n"
            "   - favor + neutral + unfavor MUST sum to exactly 1.0\n"
            "   - If data is missing or unclear, push weight into neutral\n\n"
            "3. Examples of correct triplets:\n"
            "   - High foot traffic: favor=0.8, neutral=0.2, unfavor=0.0\n"
            "   - 10 competing restaurants nearby: favor=0.0, neutral=0.3, unfavor=0.7\n"
            "   - Parking data unknown: favor=0.2, neutral=0.6, unfavor=0.2\n\n"
            "4. Sort parameters by favor descending.\n\n"
            "Return a JSON array with this exact shape:\n"
            '[\n'
            '  {\n'
            '    "parameter": "parameter name",\n'
            '    "value": "what you found in the search results",\n'
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

            # Normalize each triplet to guarantee sum = 1.0
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

        print(f"\n[Step 3] Extracted {len(parameters)} parameters with IF triplets")
        return {"parameters": parameters}
    
    # STEP 4: Generate decision tree — weights only, IF blank on every node
    def generate_decision_tree(state: AgentState) -> dict:
        params_json = json.dumps(state["parameters"], indent=2)
        system = SystemMessage(content=(
            "You are a decision analysis expert.\n\n"
            "Build a weighted decision tree from the given parameters.\n\n"
            "Structure rules:\n"
            "- Root node = the final decision\n"
            "- Group related parameters under intermediate parent nodes\n"
            "  (e.g. 'Location Factors', 'Competition', 'Economic Factors')\n"
            "- Leaf nodes = individual parameters\n"
            "- 'weight': importance of this node among its siblings (siblings must sum to 1.0)\n"
            "- Do NOT set favor/neutral/unfavor on ANY node — leave them all as 0.0\n\n"
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
            '      "weight": 0.4,\n'
            '      "favor": 0.0, "neutral": 0.0, "unfavor": 0.0,\n'
            '      "children": [\n'
            "        {\n"
            '          "id": "leaf_1",\n'
            '          "label": "Parameter Name",\n'
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
        response = model.invoke([system, human])

        try:
            raw = response.content.strip().replace("```json", "").replace("```", "")
            tree = json.loads(raw)
        except json.JSONDecodeError:
            tree = {}

        print(f"\n[Step 4] Decision tree structure generated (IF blank)")
        print(json.dumps(tree, indent=2))
        return {"decision_tree": tree}

    # STEP 5: Score IF triplets on leaf nodes only, using parameters data
    def score_leaf_if(state: AgentState) -> dict:
        tree_json  = json.dumps(state["decision_tree"], indent=2)
        params_json = json.dumps(state["parameters"], indent=2)

        system = SystemMessage(content=(
            "You are a decision analyst using the Italian Flag (IF) method.\n\n"
            "You will receive a decision tree (all IF values are 0.0) and a list of "
            "scored parameters.\n\n"
            "Your job: fill in favor, neutral, unfavor ONLY on LEAF nodes.\n\n"
            "Rules:\n"
            "- Match each leaf by its label to the corresponding parameter\n"
            "- Assign a triplet (favor, neutral, unfavor) that reflects how that "
            "parameter influences the decision\n"
            "- favor + neutral + unfavor must sum to exactly 1.0 for each leaf\n"
            "- If data is missing or unclear, push weight into neutral\n"
            "- DO NOT touch intermediate or root nodes — leave their IF as 0.0\n"
            "- DO NOT change any weights\n"
            "- Return the complete tree JSON with only leaf IF values filled in\n\n"
            "Examples of correct leaf triplets:\n"
            "  High foot traffic: favor=0.8, neutral=0.2, unfavor=0.0\n"
            "  10 competing restaurants: favor=0.0, neutral=0.3, unfavor=0.7\n"
            "  Unknown parking: favor=0.2, neutral=0.6, unfavor=0.2\n\n"
            "Return ONLY valid JSON. No markdown, no explanation."
        ))
        human = HumanMessage(content=(
            f"Decision question: {state['original_question']}\n\n"
            f"Decision tree:\n{tree_json}\n\n"
            f"Parameters:\n{params_json}"
        ))
        response = model.invoke([system, human])

        try:
            raw = response.content.strip().replace("```json", "").replace("```", "")
            tree = json.loads(raw)
        except json.JSONDecodeError:
            tree = state["decision_tree"]  # fallback: keep unscored tree

        print(f"\n[Step 5] IF triplets scored on leaf nodes")
        print(json.dumps(tree, indent=2))
        return {"decision_tree": tree}


    # STEP 6: Send tree to MCP tool — propagates IF from leaves to root
    async def calculate_tree(state: AgentState) -> dict:
        tree_str = json.dumps(state["decision_tree"])
        result = await mcp_client.call_tool(
            "process_decision_tree",
            arguments={"tree_structure": tree_str}
        )
        calculated = json.loads(result.content[0].text)
        root = calculated
        print(f"\n[Step 6] IF propagated to root")
        print(f"         favor={root.get('favor','?')}  "
            f"neutral={root.get('neutral','?')}  "
            f"unfavor={root.get('unfavor','?')}")
        return {"decision_tree": calculated}
    
    # STEP 7: Present final results
    def present_results(state: AgentState) -> dict:     # rounded for display only, 4 digits in actual data
        def if_bar(favor, neutral, unfavor, width=40):
            g = round(favor   * width)
            w = round(neutral * width)
            r = round(unfavor * width)
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

        # Extract all leaf nodes from the calculated tree
        def collect_leaves(node):
            if not node.get("children"):
                return [node]
            leaves = []
            for child in node.get("children", []):
                leaves.extend(collect_leaves(child))
            return leaves

        tree = state["decision_tree"]
        leaves = collect_leaves(tree)

        print(f"\n{'='*65}")
        print(f" LOCATION ANALYSIS — Italian Flag Method")
        print(f" {state['original_question']}")
        print(f"{'='*65}\n")

        print(f"  {'PARAMETER':<26}  {'FAV':>5} {'NEU':>5} {'UNF':>5}  FLAG")
        print(f"  {'-'*61}")

        for leaf in leaves:
            f = leaf.get("favor",   0.0)
            n = leaf.get("neutral", 0.0)
            u = leaf.get("unfavor", 0.0)
            bar = if_bar(f, n, u)
            label = if_label(f, u)
            # match reasoning from original parameters by label
            param = next((p for p in state["parameters"]
                        if p["parameter"].lower() in leaf["label"].lower()
                        or leaf["label"].lower() in p["parameter"].lower()), {})
            print(f"  {leaf['label']:<26}  {f:>5.2f} {n:>5.2f} {u:>5.2f}  {bar}  {label}")
            if param.get("value"):
                print(f"    ↳ {param['value']}")
            if param.get("reasoning"):
                print(f"    ↳ {param['reasoning']}\n")

        # Root node
        rf = tree.get("favor",   0.0)
        rn = tree.get("neutral", 0.0)
        ru = tree.get("unfavor", 0.0)

        print(f"{'='*65}")
        print(f" DECISION TREE RESULT")
        print(f"{'='*65}")
        print(f"  Favor    (green)  {rf:.3f}  {if_bar(rf, 0, 0, width=40)}")
        print(f"  Neutral  (white)  {rn:.3f}  {if_bar(0, rn, 0, width=40)}")
        print(f"  Unfavor  (red)    {ru:.3f}  {if_bar(0, 0, ru, width=40)}")
        print(f"\n  Full flag:  {if_bar(rf, rn, ru, width=50)}")
        print(f"\n  VERDICT: {if_label(rf, ru)}")
        print(f"{'='*65}\n")
        return {}
    
    return (
        reword_query,
        run_search,
        extract_and_score_parameters,
        generate_decision_tree,
        score_leaf_if,
        calculate_tree,
        present_results,
    )

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

        (
            reword_query,
            run_search,
            extract_and_score_parameters,
            generate_decision_tree,
            score_leaf_if,
            calculate_tree,
            present_results,
        ) = make_nodes(model, mcp_client, search_tool)

        workflow = StateGraph(AgentState)

        workflow.add_node("reword_query", reword_query)
        workflow.add_node("run_search", run_search)
        workflow.add_node("extract_and_score_parameters", extract_and_score_parameters)
        workflow.add_node("generate_decision_tree", generate_decision_tree)
        workflow.add_node("score_leaf_if", score_leaf_if)
        workflow.add_node("calculate_tree", calculate_tree)
        workflow.add_node("present_results", present_results)

        workflow.add_edge(START, "reword_query")
        workflow.add_edge("reword_query", "run_search")
        workflow.add_edge("run_search", "extract_and_score_parameters")
        workflow.add_edge("extract_and_score_parameters", "generate_decision_tree")
        workflow.add_edge("generate_decision_tree", "score_leaf_if")
        workflow.add_edge("score_leaf_if", "calculate_tree")
        workflow.add_edge("calculate_tree", "present_results")
        workflow.add_edge("present_results", END)

        app = workflow.compile()

        await app.ainvoke({
            "messages": [HumanMessage(content=question)],
            "original_question": question,
            "search_query": "",
            "search_results": "",
            "parameters": [],
            "decision_tree": {},
        })


if __name__ == "__main__":
    asyncio.run(run_agent(
        "Is opening a restaurant in Via Calzaiuoli 50 in Florence a good idea?"
    ))