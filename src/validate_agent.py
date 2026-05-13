import asyncio
import os
import json
import logging
import traceback
from dotenv import load_dotenv
from fastmcp import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from builder import create_agent_app

logging.basicConfig(level=logging.INFO)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

load_dotenv()

WATCH_LIST = ["TC_01_Address", "TC_08_NonExistentPOI"]

async def run_full_validation(test_id: str, question: str) -> bool:
    print(f"\n" + "═"*80)
    print(f" VALIDATING TEST CASE: {test_id}")
    print(f" QUESTION: {question}")
    print("═"*80)

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

        print(f"\n✅ {test_id} COMPLETED SUCCESSFULLY")
        return True

    except Exception as e:
        print(f"\n❌ {test_id} FAILED — {type(e).__name__}: {e}")
        # traceback.print_exc()
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
            "id": "TC_02_DifferentAddress",
            "q": "conviene aprire un ristorante in Piazza della Repubblica a Firenze?"
        },
        {
            "id": "TC_03_DifferentCity",
            "q": "conviene aprire un ristorante in Via Roma 270 a Pontedera, Pisa?"
        },
        {
            "id": "TC_04_OtherCategory",
            "q": "conviene aprire un hotel a Firenze in Via Calzaiuoli 50?"
        },
        {
            "id": "TC_05_POI",
            "q": "conviene aprire una gelateria vicino a Gelatando a Scandicci, Firenze?"
        },
        {
            "id": "TC_06_DifferentSyntax",
            "q": "sarebbe redditizio avviare una pasticceria in Via Calzaiuoli 50 a Firenze"
        },
        {
            "id": "TC_07_NonExistentAddress",
            "q": "conviene aprire un ristorante in Corso Como 100 a Milano?",
            "expect_fail": True,
        },
        {
            "id": "TC_08_NonExistentPOI",
            "q": "conviene aprire un ristorante vicino al Colosseo a Roma?",
            "expect_fail": True,
        },
    ]

    summary = []
    for test in test_suite:
        expect_fail  = test.get("expect_fail", False)
        actual_success = await run_full_validation(test["id"], test["q"])
        passed = (actual_success and not expect_fail) or (not actual_success and expect_fail)
        summary.append((test["id"], passed, expect_fail, actual_success))

    total  = len(summary)
    passed = sum(1 for _, p, _, _ in summary if p)

    print("\n" + "█"*60)
    print(f"{'TEST CASE ID':<35} | {'RESULT':<6} | EXPECTED")
    print("-" * 60)
    for tid, p, exp_fail, actual in summary:
        color  = "\033[92m" if p else "\033[91m"
        status = "PASS" if p else "FAIL"
        note   = "Expected Fail" if exp_fail else "Expected Success"
        print(f"{tid:<35} | {color}{status}\033[0m   | {note}")

    print("-" * 60)
    ratio_color = "\033[92m" if passed == total else "\033[91m"
    print(f"  Success ratio: {ratio_color}{passed}/{total} ({100*passed//total}%)\033[0m")
    print("█"*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())