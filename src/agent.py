import os
from nodes import *
from fastmcp import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from builder import create_agent_app

logging.basicConfig(level=logging.INFO)


def route_after_parse(state: AgentState) -> str:
    return "plan_leaf_scoring" if state.get("tree_reused") else "reword_query"

async def run_agent(question: str):

    mcp_client = Client("http://localhost:8000/mcp")
    await mcp_client.__aenter__()

    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        app = create_agent_app(model, mcp_client)

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