import asyncio
import json
from pathlib import Path
import sys

# Ensure project root is on sys.path so `from engine import ...` works
# when this script is executed directly (python scripts/evaluate_test_set.py).
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from typing import Any, Dict, List

from engine import SujudSenseEngine


def normalize_expected_outcome(value: str) -> str:
    return value.strip().lower()


def stage_result_check(expected: str, stage_outcome: Dict[str, Any]) -> bool:
    if expected == "vector_in_scope":
        return stage_outcome["vector_pass"] is True
    if expected == "vector_out_of_scope":
        return stage_outcome["vector_pass"] is False
    if expected == "intent_both_true":
        return stage_outcome["intent_pass"] is True

    intent = stage_outcome["intent"]
    if expected == "intent_prayer_only":
        return intent["is_prayer_related"] and not intent["has_medical_or_mobility_context"]
    if expected == "intent_medical_only":
        return not intent["is_prayer_related"] and intent["has_medical_or_mobility_context"]
    if expected == "intent_neither":
        return not intent["is_prayer_related"] and not intent["has_medical_or_mobility_context"]
    return False


async def evaluate_query(engine: SujudSenseEngine, query: str, expected: str) -> Dict[str, Any]:
    result = {
        "query": query,
        "expected": expected,
        "passed": False,
        "details": {},
    }

    if expected == "block":
        result["details"] = {"hardcoded_block": engine.is_blocked_by_hardcoded_policy(query)}
        result["passed"] = result["details"]["hardcoded_block"] is True
        return result

    if expected == "pass":
        result["details"] = {"hardcoded_block": engine.is_blocked_by_hardcoded_policy(query)}
        result["passed"] = result["details"]["hardcoded_block"] is False
        return result

    if expected == "capability_trigger":
        result["details"] = {"capability_trigger": engine.is_capability_query(query)}
        result["passed"] = result["details"]["capability_trigger"] is True
        return result

    if expected == "capability_not_trigger":
        result["details"] = {"capability_trigger": engine.is_capability_query(query)}
        result["passed"] = result["details"]["capability_trigger"] is False
        return result

    if expected in {"vector_in_scope", "vector_out_of_scope", "intent_both_true", "intent_prayer_only", "intent_medical_only", "intent_neither"}:
        stage_outcome = await engine.evaluate_stages(query, [])
        result["details"] = {
            "standalone_query": stage_outcome["standalone_query"],
            "vector_score": stage_outcome["vector_score"],
            "vector_pass": stage_outcome["vector_pass"],
            "intent": stage_outcome["intent"],
            "intent_pass": stage_outcome["intent_pass"],
        }
        result["passed"] = stage_result_check(expected, stage_outcome)
        return result

    result["details"] = {"error": "unknown expected outcome"}
    return result


def load_test_set(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    test_cases = []
    for stage, groups in data.items():
        for group_name, items in groups.items():
            for item in items:
                test_cases.append(
                    {
                        "id": item["id"],
                        "query": item["query"],
                        "expected_outcome": normalize_expected_outcome(item["expected_outcome"]),
                        "stage": stage,
                        "group": group_name,
                    }
                )

    return test_cases


async def main() -> None:
    engine = SujudSenseEngine()
    await engine.initialize()

    test_cases = load_test_set(Path("tests/test_set.json"))
    failures = []

    for case in test_cases:
        result = await evaluate_query(engine, case["query"], case["expected_outcome"])
        if not result["passed"]:
            failures.append({**case, **result})
            print(f"FAIL: {case['id']} ({case['expected_outcome']}) -> {result['details']}")
        else:
            print(f"PASS: {case['id']} ({case['expected_outcome']})")

    print("\nSummary:")
    print(f"Total: {len(test_cases)}")
    print(f"Failures: {len(failures)}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
