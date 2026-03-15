# eval/evaluate.py
import sys
import json
from pathlib import Path

# Make src/ importable from eval/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import RAGPipeline


def load_golden_dataset(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def check_faithfulness(answer: str, expected: str) -> bool:
    """
    Simple faithfulness check — did the answer contain
    the key facts from the expected answer?
    In Phase 3 this is rule-based; in production you'd use
    an LLM judge or Ragas for this.
    """
    # Refusal check — both should refuse or both should answer
    answer_refused = "cannot answer" in answer.lower()
    expected_refused = "cannot answer" in expected.lower()
    if answer_refused != expected_refused:
        return False

    # For non-refusals, check key terms from expected appear in answer
    if not expected_refused:
        key_terms = [
            word.lower() for word in expected.split()
            if len(word) > 5  # only meaningful words
        ]
        matches = sum(1 for term in key_terms if term in answer.lower())
        match_ratio = matches / max(len(key_terms), 1)
        return match_ratio >= 0.3   # 30% key term overlap = faithful

    return True


def run_evaluation(
    pipeline: RAGPipeline,
    dataset: list,
    threshold: float = 0.7   # 70% pass rate required
) -> dict:
    """
    Run all questions through the pipeline, score against
    golden answers, return pass/fail + metrics.
    """
    print(f"\n=== Evaluation ({len(dataset)} questions) ===\n")

    results = []
    passed = 0

    for item in dataset:
        result = pipeline.query(item["question"])
        answer = result["answer"]
        faithful = check_faithfulness(answer, item["expected_answer"])

        if faithful:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected": item["expected_answer"],
            "actual": answer,
            "faithful": faithful,
            "status": status,
        })

        print(f"  [{status}] {item['id']}: {item['question'][:50]}...")
        if not faithful:
            print(f"         Expected: {item['expected_answer'][:80]}...")
            print(f"         Got:      {answer[:80]}...")

    score = passed / len(dataset)
    passed_threshold = score >= threshold

    print(f"\n  Score     : {passed}/{len(dataset)} ({score:.0%})")
    print(f"  Threshold : {threshold:.0%}")
    print(f"  Result    : {'PASSED' if passed_threshold else 'FAILED'}")

    return {
        "score": score,
        "passed": passed,
        "total": len(dataset),
        "threshold": threshold,
        "passed_threshold": passed_threshold,
        "results": results,
    }


if __name__ == "__main__":
    # Boot the pipeline
    rag = RAGPipeline()
    rag.index(
        urls=["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"]
    )

    # Load golden dataset
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    dataset = load_golden_dataset(str(dataset_path))

    # Run evaluation
    report = run_evaluation(rag, dataset, threshold=0.7)

    # Save report
    report_path = Path(__file__).parent / "eval_report.json"
    with open(report_path, "w") as f:
        # Don't serialize full results to keep report clean
        summary = {k: v for k, v in report.items() if k != "results"}
        json.dump(summary, f, indent=2)
    print(f"\n  Report saved to {report_path}")

    # Exit with error code if below threshold — this is what CI uses
    if not report["passed_threshold"]:
        print("\n  CI: BUILD FAILED — quality below threshold")
        sys.exit(1)
    else:
        print("\n  CI: BUILD PASSED")
        sys.exit(0)