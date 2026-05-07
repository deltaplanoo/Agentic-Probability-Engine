from langgraph.graph import StateGraph, START, END
from nodes import make_nodes, AgentState

def route_after_parse(state: AgentState) -> str:
    return "plan_leaf_scoring" if state.get("tree_reused") else "reword_query"

def create_agent_app(model, mcp_client):
    """
    Create and compile agent's graph. 
    This is the single source for the workflow structure.
    """
    (
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
    ) = make_nodes(model, mcp_client)

    workflow = StateGraph(AgentState)

    workflow.add_node("parse_question",               parse_question)
    workflow.add_node("reword_query",                 reword_query)
    workflow.add_node("run_search",                   run_search)
    workflow.add_node("extract_and_score_parameters", extract_and_score_parameters)
    workflow.add_node("generate_decision_trees",      generate_decision_trees)
    workflow.add_node("pick_best_tree",               pick_best_tree)
    workflow.add_node("annotate_and_save_template",   annotate_and_save_template)
    workflow.add_node("plan_leaf_scoring",            plan_leaf_scoring)
    workflow.add_node("score_leaf_if",                score_leaf_if)
    workflow.add_node("calculate_tree",               calculate_tree)
    workflow.add_node("present_results",              present_results)

    workflow.add_edge(START, "parse_question")
    workflow.add_conditional_edges(
        "parse_question",
        route_after_parse,
        {
            "reword_query":      "reword_query",
            "plan_leaf_scoring": "plan_leaf_scoring",
        }
    )
    workflow.add_edge("reword_query",                 "run_search")
    workflow.add_edge("run_search",                   "extract_and_score_parameters")
    workflow.add_edge("extract_and_score_parameters", "generate_decision_trees")
    workflow.add_edge("generate_decision_trees",      "pick_best_tree")
    workflow.add_edge("pick_best_tree",               "annotate_and_save_template")
    workflow.add_edge("annotate_and_save_template",   "plan_leaf_scoring")
    workflow.add_edge("plan_leaf_scoring",            "score_leaf_if")
    workflow.add_edge("score_leaf_if",                "calculate_tree")
    workflow.add_edge("calculate_tree",               "present_results")
    workflow.add_edge("present_results",              END)

    return workflow.compile()