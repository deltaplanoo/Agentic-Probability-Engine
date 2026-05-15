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

async def run_full_validation(test_id: str, question: str) -> tuple[bool, str | None]:
    """Returns (actual_success, if_verdict) where if_verdict is 'favorable'/'unfavorable'/'uncertain'/None"""
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

        result = await app.ainvoke({
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

        tree = result.get("decision_tree", {})
        f = tree.get("favor",   0.0)
        n = tree.get("neutral", 0.0)
        u = tree.get("unfavor", 0.0)

        if f > u and f > 0.45:
            verdict = "favorable"
        elif u > f and u > 0.45:
            verdict = "unfavorable"
        elif f > 0.35 and u < 0.15:
            verdict = "favorable"
        elif u > 0.35 and f < 0.15:
            verdict = "unfavorable"
        else:
            verdict = "uncertain"

        print(f"\n✅ {test_id} COMPLETED — IF verdict: {verdict} (f={f:.3f} n={n:.3f} u={u:.3f})")
        return True, verdict

    except Exception as e:
        print(f"\n❌ {test_id} FAILED — {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, None

    finally:
        await mcp_client.__aexit__(None, None, None)

async def main():
    test_suite = [
        {
            "id": "TC_01_Address",
            "q": "conviene aprire un ristorante in Via Calzaiuoli 50 a Firenze?",
            "expect_fail": False,
            "if": "favorable"
        },
        {
            "id": "TC_03_DifferentCity",
            "q": "conviene aprire un dentista in Via Roma 270 a Pontedera, Pisa?",
            "expect_fail": False,
            "if": "favorable"
        },
        {
            "id": "TC_04_OtherCategory",
            "q": "conviene aprire un hotel in Piazza San Michele a Lucca?",
            "expect_fail": False,
            "if": "favorable"
        },
        {
            "id": "TC_05_OtherMacrocategory",
            "q": "sarebbe una buona idea aprire una raffineria di petrolio a Firenze in Via Calzaiuoli 50?",
            "expect_fail": False,
            "if": "unfavorable"
        },
        {
            "id": "TC_06_POI",
            "q": "conviene aprire una gelateria vicino a Gelatando a Scandicci, Firenze?",
            "expect_fail": False,
            "if": "favorable"
        },
        {
            "id": "TC_07_OtherCategoryUnfavorable",
            "q": "sarebbe una buona idea aprire un aquario a Santa Brigida a Pontassieve?",
            "expect_fail": False,
            "if": "unfavorable"
        },
        {
            "id": "TC_08_NonExistentAddress",
            "q": "conviene aprire una concessionaria in Corso Como 100 a Milano?",
            "expect_fail": True,
        },
        {
            "id": "TC_09_NonExistentPOI",
            "q": "conviene aprire un ristorante vicino al Colosseo a Roma?",
            "expect_fail": True,
        },
    ]

    summary = []
    for test in test_suite:
        expect_fail    = test.get("expect_fail", False)
        expected_if    = test.get("if")
        actual_success, actual_verdict = await run_full_validation(test["id"], test["q"])

        exec_passed = (actual_success and not expect_fail) or (not actual_success and expect_fail)

        # IF correctness (0, 0.5, o 1)
        if_score = None
        if not expect_fail and expected_if and actual_success:
            if actual_verdict == expected_if:
                if_score = 1.0
            elif "uncertain" in actual_verdict.lower() and expected_if in ["favorable", "unfavorable"]:
                if_score = 0.5
            else:
                if_score = 0.0

        summary.append((test["id"], exec_passed, if_score, expect_fail, expected_if, actual_verdict))

    total        = len(summary)
    exec_passed  = sum(1 for _, ep, _, _, _, _ in summary if ep)
    
    if_tests     = [s for s in summary if s[2] is not None]
    if_correct_sum = sum(s[2] for s in if_tests)

    print("\n" + "█"*75)
    print(f"{'TEST CASE ID':<35} | {'EXEC':<6} | {'IF EXPECTED':<14} | {'IF ACTUAL':<14} | {'SCORE'}")
    print("-" * 75)
    for tid, ep, score, exp_fail, expected_if, actual_verdict in summary:
        exec_color  = "\033[92m" if ep else "\033[91m"
        exec_status = "PASS" if ep else "FAIL"
        exp_if_str  = expected_if or ("FAIL" if exp_fail else "—")
        act_if_str  = actual_verdict or "—"
        
        if score is None:
            score_str = "—"
            score_color = "\033[0m"
        else:
            score_str = f"{score}"
            score_color = "\033[92m" if score == 1.0 else ("\033[93m" if score == 0.5 else "\033[91m")
            
        print(f"{tid:<35} | {exec_color}{exec_status}\033[0m   | {exp_if_str:<14} | {act_if_str:<14} | {score_color}{score_str}\033[0m")

    print("-" * 75)
    exec_color = "\033[92m" if exec_passed == total else "\033[91m"
    print(f"  Execution pass rate : {exec_color}{exec_passed}/{total} ({100*exec_passed//total}%)\033[0m")

    if if_tests:
        # Calcolo percentuale basato sulla somma dei punteggi
        accuracy_perc = (if_correct_sum / len(if_tests)) * 100
        if_color = "\033[92m" if accuracy_perc >= 80 else ("\033[93m" if accuracy_perc >= 50 else "\033[91m")
        print(f"  IF accuracy score   : {if_color}{if_correct_sum}/{len(if_tests)} ({accuracy_perc:.1f}%)\033[0m")
    else:
        print(f"  IF accuracy rate    : — (no completed IF tests)")

    print("█"*75 + "\n")


if __name__ == "__main__":
    asyncio.run(main())