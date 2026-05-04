import asyncio
import os
import json
import traceback
from dotenv import load_dotenv
from fastmcp import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from nodes import make_nodes, AgentState

load_dotenv()

async def run_full_validation(test_id: str, question: str):
    """
    Executes the complete 11-step agentic workflow manually to validate 
    changes in nodes.py and MCP tool connectivity.
    """
    print(f"\n" + "═"*80)
    print(f" VALIDATING TEST CASE: {test_id}")
    print(f" QUESTION: {question}")
    print("═"*80)

    mcp_client = Client("http://localhost:8000/mcp")
    
    try:
        await mcp_client.__aenter__()
        
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
            plan_leaf_scoring,
            score_leaf_if,
            calculate_tree,
            present_results,
        ) = make_nodes(model, mcp_client)

        state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "original_question": question,
            "decision_type": "",
            "variables": {},
            "search_query": "",
            "search_results": "",
            "parameters": [],
            "candidate_trees": [],
            "decision_tree": {},
            "tree_reused": False,
        }

        # TEST FLOW

        print(f"\n[{test_id}] Step 1: Parsing question and geocoding...")
        state.update(await parse_question(state))

        if not state.get("tree_reused"):
            print(f"[{test_id}] Step 2: Optimizing search query...")
            state.update(reword_query(state))

            print(f"[{test_id}] Step 3: Global web search...")
            state.update(await run_search(state))

            print(f"[{test_id}] Step 4: Extracting parameters...")
            state.update(extract_and_score_parameters(state))

            print(f"[{test_id}] Step 5a: Generating candidate trees...")
            state.update(await generate_decision_trees(state))

            print(f"[{test_id}] Step 5b: Picking best candidate...")
            state.update(pick_best_tree(state))

            print(f"[{test_id}] Step 6: Parameterizing and saving template...")
            state.update(annotate_and_save_template(state))
        else:
            print(f"[{test_id}] (Steps 2-6 skipped: Reusing template from DB)")

        print(f"[{test_id}] Step 6.5: Planning scoring strategy (Taxonomy lookup)...")
        state.update(await plan_leaf_scoring(state))

        print(f"[{test_id}] Step 7: Scoring leaves (Snap4City POIs + Web Sentiment)...")
        state.update(await score_leaf_if(state))

        print(f"[{test_id}] Step 8: Propagating probabilities to root...")
        state.update(await calculate_tree(state))

        print(f"[{test_id}] Step 9: Rendering final report...")
        present_results(state)

        print(f"\n✅ TEST CASE {test_id} COMPLETED SUCCESSFULLY")
        return True

    except Exception as e:
        print(f"\n❌ TEST CASE {test_id} FAILED")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("-" * 40)
        traceback.print_exc()
        return False

    finally:
        await mcp_client.__aexit__(None, None, None)

async def main():
    test_suite = [
        {
            "id": "TC_01_Address",
            "q": "conviene aprire un ristorante in Via Calzaiuoli 50 a Firenze?"
        },
        {
            "id": "TC_01_DifferentAddress",
            "q": "conviene aprire un ristorante in ?"
        },
        {
            "id": "TC_02_OtherCategory",
            "q": "conviene aprire un hotel in Via Calzaiuoli 50 a Firenze?"
        },
        {
            "id": "TC_03_POI",
            "q": "conviene aprire una gelateria vicino a Gelatando a Scandicci, Firenze?"
        },
        {
            "id": "TC_04_Complex",
            "q": "conviene aprire una pasticceria in Via Calzaiuoli 50 a Firenze?"
        },
        {
            "id": "TC_05_NonExistentAddress", #FIXME: find a truly non-existent address to trigger geocoding failure
            "q": "conviene aprire un ristorante in via Almeffini 12 a Firenze?"
        },
        {
            "id": "TC_06_NonExistentPOI",
            "q": "conviene aprire un ristorante vicino al Colosseo a Roma?"
        },
    ]

    summary = []
    for test in test_suite:
        success = await run_full_validation(test["id"], test["q"])
        summary.append((test["id"], "PASS" if success else "FAIL"))

    # Final Summary Table
    print(f"{'TEST CASE ID':<35} | {'STATUS':<10}")
    print("-" * 50)
    for tid, status in summary:
        color = "\033[92m" if status == "PASS" else "\033[91m"
        print(f"{tid:<35} | {color}{status}\033[0m")
    print("█"*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())