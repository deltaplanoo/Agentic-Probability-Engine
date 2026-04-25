import asyncio
import json
import re
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from session_store import save_template, load_template, print_templates

load_dotenv()

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:          Annotated[list[BaseMessage], add_messages]
    original_question: str
    decision_type:     str
    variables:         dict
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
                node[key] = node[key].replace(f"{{{var_name}}}", str(var_value))
    
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

def update_leaf_scoring_strategy(node: dict, strategies: dict) -> dict:
    """Recursively write scoring_strategy into each leaf whose id appears in strategies."""
    node = dict(node)
    if not node.get("children"):
        if node["id"] in strategies:
            node["scoring_strategy"] = strategies[node["id"]]
        return node
    node["children"] = [
        update_leaf_scoring_strategy(c, strategies) for c in node["children"]
    ]
    return node

# ── Nodes ─────────────────────────────────────────────────────────────────────

def make_nodes(model, mcp_client):

    # STEP 1: Parse question — extract decision_type + variables, check DB
    async def parse_question(state: AgentState) -> dict:
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

        update = {
            "decision_type": decision_type,
            "variables":     variables,
            "tree_reused":   False,
        }

        if "address" in variables:
            address = variables["address"]
            print(f"[Step 1] Geocoding via address_search_location: '{address}'")
            try:
                mcp_res  = await mcp_client.call_tool(
                    "address_search_location",
                    arguments={
                        "search":     address,
                        "logic":      "AND",
                        "excludePOI": True,
                        "maxresults": 5,
                        "lang":       "it",
                    }
                )
                geo_data = json.loads(mcp_res.content[0].text)
                error    = geo_data.get("error")
                features = geo_data.get("results") or []

                if error:
                    print(f"[Step 1] Geocoding error: {error}")
                elif features:
                    # GeoJSON coordinates are [longitude, latitude]
                    coords = features[0].get("geometry", {}).get("coordinates", [])
                    if len(coords) >= 2:
                        variables["lon"] = coords[0]
                        variables["lat"] = coords[1]
                        addr_label = features[0].get("properties", {}).get("address", address)
                        print(f"[Step 1] Coordinates found: lat={coords[1]}, lon={coords[0]}  ({addr_label})")
                    else:
                        print(f"[Step 1] Geocoding: no coordinates in first feature")
                else:
                    print(f"[Step 1] Geocoding: no results returned for '{address}'")
            except Exception as e:
                print(f"[Step 1] Geocoding failed: {e}")

        template = load_template(decision_type)
        if template:
            print(f"[Step 1] Template found for '{decision_type}' — reusing")
            injected_tree = inject_variables(template["tree"], variables)
            update["decision_tree"] = injected_tree
            update["tree_reused"]   = True
        else:
            print(f"[Step 1] No template found for '{decision_type}' — will generate")

        return update

    # STEP 2: Reword question into search query (first-run only)
    def reword_query(state: AgentState) -> dict:
        system = SystemMessage(content=(
            "You are a search query optimizer. "
            "Given a decision-making question about a physical location, "
            "rewrite it as a concise, effective web search query to find "
            "data useful for evaluating that location for the decision. "
            "Return ONLY the search query string, nothing else."
        ))
        response = model.invoke([system, HumanMessage(content=state["original_question"])])
        query = response.content.strip()
        print(f"\n[Step 2] Reworded query: {query}")
        return {"search_query": query}

    # STEP 3: Global web search (first-run only)
    async def run_search(state: AgentState) -> dict:
        mcp_res = await mcp_client.call_tool("web_search", arguments={"query": state["search_query"]})
        result = mcp_res.content[0].text
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
            "  Leaf 'Restaurant competition': search_hint = 'restaurants near {address}'\n"
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

        variable_names = list(variables.keys())

        save_template(state["decision_type"], variable_names, annotated_tree)
        print(f"\n[Step 6] Template annotated and saved for '{state['decision_type']}'")

        # re-inject variables
        injected_tree = inject_variables(annotated_tree, variables)
        return {"decision_tree": injected_tree}

    # STEP 6.5: For each leaf, explore Snap4City categories via MCP to decide the scoring tool
    async def plan_leaf_scoring(state: AgentState) -> dict:
        tree       = state["decision_tree"]
        leaves     = collect_leaves(tree)
        variables  = state["variables"]
        has_coords = "lat" in variables and "lon" in variables

        print(f"\n[Step 6.5] Planning scoring strategy for {len(leaves)} leaves...")

        # Fetch macrocategory list once — shared across all leaf planners
        macrocategories: list[str] = []
        if has_coords:
            try:
                macro_res   = await mcp_client.call_tool("get_poi_categories", arguments={})
                macro_data  = json.loads(macro_res.content[0].text)
                macrocategories = macro_data.get("macrocategories", [])
                print(f"[Step 6.5] Snap4City macrocategories available: {len(macrocategories)}")
            except Exception as e:
                print(f"[Step 6.5] Could not load macrocategories ({e}); all leaves → web_search")

        loop = asyncio.get_running_loop()

        async def plan_single_leaf(leaf: dict) -> tuple[str, dict]:
            """
            Per-leaf agentic exploration:
              1. LLM picks the most relevant macrocategory (or 'none').
              2. If a macrocategory is picked, fetch its subcategories via MCP.
              3. LLM picks the closest subcategory (or falls back to macrocategory-level, or web_search).
            """
            leaf_id    = leaf["id"]
            leaf_label = leaf["label"]
            hint       = leaf.get("search_hint", leaf_label)

            # ── Phase 1: decide if POI data is relevant + which macrocategory ──
            if not has_coords or not macrocategories:
                print(f"  [{leaf_label}] → web_search (no coords)")
                return leaf_id, {"tool": "web_search"}

            sys_macro = SystemMessage(content=(
                "You are a data-source router for a location-decision system.\n\n"
                "Decide whether the given decision leaf can be evaluated by COUNTING nearby physical "
                "Places/POIs in a city using the Snap4City database.\n\n"
                "Rules:\n"
                "  - Answer 'NONE' if the leaf is about trends, prices, demographics, regulations, "
                "sentiment or anything that cannot be answered by a POI count.\n"
                "  - Otherwise, respond with the SINGLE most relevant macrocategory name from the list, "
                "EXACTLY as written — do not invent names.\n\n"
                f"Available macrocategories:\n{chr(10).join(macrocategories)}\n\n"
                "Return ONLY the macrocategory name or the word NONE. No explanation."
            ))
            human_macro = HumanMessage(content=(
                f"Decision: {state['original_question']}\n"
                f"Leaf label: {leaf_label}\n"
                f"Search hint: {hint}"
            ))

            resp_macro = await loop.run_in_executor(
                None, lambda: model.invoke([sys_macro, human_macro])
            )
            chosen_macro = resp_macro.content.strip().strip('"').strip("'")

            if chosen_macro.upper() == "NONE" or chosen_macro not in macrocategories:
                print(f"  [{leaf_label}] → web_search (no matching macrocategory)")
                return leaf_id, {"tool": "web_search"}

            # ── Phase 2: fetch subcategories for the chosen macrocategory ──
            try:
                sub_res   = await mcp_client.call_tool(
                    "get_poi_categories", arguments={"macro_category": chosen_macro}
                )
                sub_data  = json.loads(sub_res.content[0].text)
                subcats   = sub_data.get("categories", [])
            except Exception as e:
                print(f"  [{leaf_label}] → snap4city macrocategory '{chosen_macro}' (subcats fetch failed: {e})")
                return leaf_id, {"tool": "snap4city", "snap4city_type": "macrocategory", "snap4city_cat": chosen_macro}

            if not subcats:
                print(f"  [{leaf_label}] → snap4city macrocategory '{chosen_macro}' (no subcategories)")
                return leaf_id, {"tool": "snap4city", "snap4city_type": "macrocategory", "snap4city_cat": chosen_macro}

            # ── Phase 3: pick closest subcategory (or fall back to macro) ──
            sys_cat = SystemMessage(content=(
                f"You are matching a decision leaf to a Snap4City POI subcategory.\n\n"
                f"Macrocategory chosen: {chosen_macro}\n"
                f"Available subcategories:\n{chr(10).join(subcats)}\n\n"
                "Pick the SINGLE subcategory that best represents the physical venues to count "
                "for this leaf, EXACTLY as written.\n"
                "If none of the subcategories is a good match, respond with NONE.\n"
                "Return ONLY the subcategory name or NONE. No explanation."
            ))
            human_cat = HumanMessage(content=(
                f"Decision: {state['original_question']}\n"
                f"Leaf label: {leaf_label}\n"
                f"Search hint: {hint}"
            ))

            resp_cat = await loop.run_in_executor(
                None, lambda: model.invoke([sys_cat, human_cat])
            )
            chosen_cat = resp_cat.content.strip().strip('"').strip("'")

            if chosen_cat.upper() == "NONE" or chosen_cat not in subcats:
                # valid macro but no precise subcategory → use macro api call
                print(f"  [{leaf_label}] → snap4city macrocategory '{chosen_macro}'")
                return leaf_id, {"tool": "snap4city", "snap4city_type": "macrocategory", "snap4city_cat": chosen_macro}

            print(f"  [{leaf_label}] → snap4city category '{chosen_cat}' (under {chosen_macro})")
            return leaf_id, {"tool": "snap4city", "snap4city_type": "category", "snap4city_cat": chosen_cat}

        # run all leaf planners in parallel
        results = await asyncio.gather(*[plan_single_leaf(leaf) for leaf in leaves])
        strategies = {leaf_id: strat for leaf_id, strat in results}

        updated_tree = update_leaf_scoring_strategy(tree, strategies)
        return {"decision_tree": updated_tree}

    # STEP 7: Score IF on each leaf using the tool chosen in Step 6.5
    # 2 scoring paths:
    #   - web_search: sentiment analysis on textual results
    #   - snap4city:  POI count vs threshold comparison
    POI_COUNT_THRESHOLD = 15 # FIXME: make it dynamic per category
    async def score_leaf_if(state: AgentState) -> dict:
        tree      = state["decision_tree"]
        leaves    = collect_leaves(tree)
        variables = state["variables"]
        lat = variables.get("lat")
        lon = variables.get("lon")

        print(f"\n[Step 7] Scoring {len(leaves)} leaves individually...")

        loop = asyncio.get_running_loop()

        async def score_single_leaf(leaf: dict) -> tuple[str, float, float, float]:
            leaf_id    = leaf["id"]
            leaf_label = leaf["label"]
            hint       = leaf.get("search_hint", leaf_label)

            strategy  = leaf.get("scoring_strategy", {"tool": "web_search"})
            tool      = strategy.get("tool", "web_search")

            # ── Snap4City ──────────────────────────────────────
            if tool == "snap4city" and lat and lon:
                s4c_type = strategy.get("snap4city_type", "category")
                s4c_term = strategy.get("snap4city_cat", leaf_label)
                print(f"  [{leaf_label}] Snap4City {s4c_type}: '{s4c_term}'")

                try:
                    mcp_res = await mcp_client.call_tool(
                        "get_poi_nearby",
                        arguments={
                            "search_term": s4c_term,
                            "lat": lat,
                            "lon": lon,
                            "max_dist_km": 0.5,
                        }
                    )
                    raw_poi = mcp_res.content[0].text.strip()
                except Exception as e:
                    print(f"    ✗ Snap4City call failed ({e}), marking neutral")
                    return (leaf_id, 0.0, 1.0, 0.0)

                if not raw_poi:
                    print(f"    ✗ Empty Snap4City result, marking neutral")
                    return (leaf_id, 0.0, 1.0, 0.0)

                # Extract integer count from the result text
                # Expected format: "Total count of '<term>' in the area: N"
                count_match = re.search(r"(\d+)", raw_poi)
                poi_count   = int(count_match.group(1)) if count_match else 0
                print(f"    POI count: {poi_count} (threshold={POI_COUNT_THRESHOLD})")

                # Ask the LLM whether a HIGH count is good or bad for this decision,
                # then compute the IF triplet from the count ratio accordingly.
                sys_ctx = SystemMessage(content=(
                    "You are a location decision analyst.\n\n"
                    "For the given decision and leaf parameter, answer with a single word:\n"
                    "  'POSITIVE' — if a HIGH count of nearby POIs is favorable for the decision\n"
                    "               (e.g. many tourists → good for a restaurant)\n"
                    "  'NEGATIVE' — if a HIGH count is unfavorable\n"
                    "               (e.g. many restaurants nearby → bad for opening a new one)\n"
                    "Return ONLY the word POSITIVE or NEGATIVE."
                ))
                human_ctx = HumanMessage(content=(
                    f"Decision: {state['original_question']}\n"
                    f"Leaf: {leaf_label} (searching for: {s4c_term})"
                ))
                resp_ctx = await loop.run_in_executor(
                    None, lambda: model.invoke([sys_ctx, human_ctx])
                )
                polarity = resp_ctx.content.strip().upper()
                is_positive = "POSITIVE" in polarity   # HIGH count → favor

                # Compute ratio: 0.0 = no POIs, 1.0 = at or above threshold
                ratio = min(poi_count / POI_COUNT_THRESHOLD, 1.0)

                if is_positive:
                    # More POIs → more favorable
                    f = round(ratio,           4)
                    u = round((1.0 - ratio),   4)
                    n = round(1.0 - f - u,     4)
                else:
                    # More POIs → more unfavorable (competition / saturation)
                    u = round(ratio,           4)
                    f = round((1.0 - ratio),   4)
                    n = round(1.0 - f - u,     4)

                # Clamp neutral to [0, 1] in case of floating-point drift
                n = max(n, 0.0)
                print(f"    polarity={polarity}  favor={f}  neutral={n}  unfavor={u}")
                return (leaf_id, f, n, u)

            # ── PATH B: Web search + sentiment analysis ──────────────────────────
            else:
                print(f"  [{leaf_label}] Web Search...")
                try:
                    mcp_res    = await mcp_client.call_tool("web_search", arguments={"query": hint})
                    web_result = mcp_res.content[0].text.strip()
                except Exception as e:
                    print(f"    ✗ Web search failed ({e}), marking neutral")
                    return (leaf_id, 0.0, 1.0, 0.0)

                if not web_result:
                    print(f"    ✗ Empty web result, marking neutral")
                    return (leaf_id, 0.0, 1.0, 0.0)

                sys_sent = SystemMessage(content=(
                    "You are a decision analyst using the Italian Flag (IF) method.\n\n"
                    "Perform a SENTIMENT ANALYSIS on the provided web search results "
                    "to evaluate how this parameter affects the given decision.\n\n"
                    "Guidelines:\n"
                    "  - 'favor'   : evidence that clearly SUPPORTS the decision succeeding\n"
                    "  - 'unfavor' : evidence that clearly OPPOSES or threatens the decision\n"
                    "  - 'neutral' : missing data, contradictory signals, or irrelevant content\n"
                    "  - The three values MUST sum to exactly 1.0\n"
                    "  - Push uncertainty into 'neutral', NOT into favor or unfavor\n\n"
                    "Return ONLY valid JSON: "
                    "{\"favor\": 0.0, \"neutral\": 0.0, \"unfavor\": 0.0, \"reasoning\": \"one sentence\"}"
                ))
                human_sent = HumanMessage(content=(
                    f"Decision: {state['original_question']}\n"
                    f"Parameter being evaluated: {leaf_label}\n\n"
                    f"Web search results:\n{web_result}"
                ))

                resp_sent = await loop.run_in_executor(
                    None, lambda: model.invoke([sys_sent, human_sent])
                )

                try:
                    raw    = resp_sent.content.strip().replace("```json", "").replace("```", "")
                    scored = json.loads(raw)
                    f = scored.get("favor",   0.0)
                    n = scored.get("neutral",  0.0)
                    u = scored.get("unfavor",  0.0)
                    total = f + n + u
                    if total > 0:
                        f, n, u = round(f/total, 4), round(n/total, 4), round(u/total, 4)
                    else:
                        f, n, u = 0.0, 1.0, 0.0
                    reasoning = scored.get("reasoning", "")
                    print(f"    favor={f}  neutral={n}  unfavor={u}  ← {reasoning}")
                except Exception:
                    f, n, u = 0.0, 1.0, 0.0

                return (leaf_id, f, n, u)

        results = await asyncio.gather(*[score_single_leaf(leaf) for leaf in leaves])

        updated_tree = tree
        for leaf_id, f, n, u in results:
            updated_tree = update_leaf_in_tree(updated_tree, leaf_id, f, n, u)

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
            bar   = if_bar(f, n, u, width=30)
            label = if_label(f, u)
            print(f"  {leaf['label'][:30]:<30} {f:>5.2f} {n:>5.2f} {u:>5.2f}  {bar}  {label}")

        rf = tree.get("favor",   0.0)
        rn = tree.get("neutral", 0.0)
        ru = tree.get("unfavor", 0.0)

        print(f"\n{'='*65}")
        print(f" DECISION TREE RESULT")
        print(f"{'='*65}")
        print(f"\n  Favor: {rf:.4f}  Neutral: {rn:.4f}  Unfavor: {ru:.4f}")
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
        plan_leaf_scoring,
        score_leaf_if,
        calculate_tree,
        present_results,
    )
