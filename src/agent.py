import os
from nodes import *
from fastmcp import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from tools_factory import get_mcp_tools


def route_after_parse(state: AgentState) -> str:
    return "score_leaf_if" if state.get("tree_reused") else "reword_query"

async def run_agent(question: str):

    mcp_client = Client("http://localhost:8000/sse")
    await mcp_client.__aenter__()

    try:
        mcp_tools = await get_mcp_tools(mcp_client)

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
        ) = make_nodes(model, mcp_client, mcp_tools)

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
        "Is opening a restaurant in Via Calzaiuoli 50 Firenze a good idea?"
    ))