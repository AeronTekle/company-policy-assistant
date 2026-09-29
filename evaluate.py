"""Run genuine measurements. Expected answers are never passed to the engine."""
import argparse
import json
import os
import random
import time
from pathlib import Path
from policy_engine import Gemini, PolicyError, METHODS, load_policies, build_index, answer, DEFAULT_MODEL, DEFAULT_EMBEDDING


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--limit", type=int, default=4, help="Small pilot first; 12 covers the full test set")
    parser.add_argument("--delay", type=float, default=6, help="Seconds between successful requests; adjust for your actual quota")
    parser.add_argument("--output", default="results/measurements.json")
    args = parser.parse_args()
    if args.delay < 0 or args.limit < 1:
        parser.error("Delay must be nonnegative and limit must be positive.")
    records = load_policies()
    tests = json.loads(Path(__file__).with_name("test_questions.json").read_text())[:args.limit]
    client = None if args.rules_only else Gemini(os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL", DEFAULT_MODEL), os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING))
    index, setup = (None, None) if args.rules_only else build_index(client, records)
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists. Choose another output path to preserve previous measurements.")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {"index_setup": setup, "results": [], "protocol": "Same questions; shuffled method order seed 42; index setup excluded; no automatic factual grading."}
    rng = random.Random(42)
    for test in tests:
        methods = ["rules"] if args.rules_only else list(METHODS)
        rng.shuffle(methods)
        for method in methods:
            try:
                result = answer(test["question"], method, records, client, index)
            except PolicyError as exc:
                result = {"method": method, "question": test["question"], "error": str(exc)}
            result["test_id"] = test["id"]
            data["results"].append(result)
            output.write_text(json.dumps(data, indent=2))
            print(test["id"], method, "ERROR" if "error" in result else "saved")
            if "error" in result:
                print(result["error"])
                return  # Preserve partial progress; avoid repeated failing/quota requests.
            if method != "rules":
                time.sleep(args.delay)
    print(f"Saved real measurements to {output}. Review against test_questions.json before writing the comparison.")


if __name__ == "__main__":
    main()
