from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from research_buckets import evaluate_buckets


RESULT_PREFIX = "BUCKET_RESULT_JSON="


def merge_bucket_results(results: list[dict]) -> dict:
    if not results:
        raise RuntimeError("inference did not evaluate popularity buckets")

    merged = {
        bucket: {
            "num_cases": None,
            "hr": {},
            "coverage": {},
        }
        for bucket in evaluate_buckets.BUCKET_NAMES
    }
    for result in results:
        for bucket in evaluate_buckets.BUCKET_NAMES:
            target = merged[bucket]
            current = result[bucket]
            value = current["num_cases"]
            if target["num_cases"] not in (None, value):
                raise ValueError(
                    f"inconsistent num_cases for popularity bucket {bucket}"
                )
            target["num_cases"] = value
            for metric in ("hr", "coverage"):
                for k, value in current[metric].items():
                    if k in target[metric] and target[metric][k] != value:
                        raise ValueError(
                            f"inconsistent {metric}@{k} for popularity bucket {bucket}"
                        )
                    target[metric][k] = value
    return merged


def run_inference(script: Path, script_args: list[str]) -> dict:
    captures = []
    original_hr = evaluate_buckets.evaluate_bucketed_hr
    original_argv = sys.argv[:]
    original_path = sys.path[:]
    original_cwd = Path.cwd()

    def capture_bucket_metrics(targets, predictions, bucket_by_item, ks):
        hr = original_hr(targets, predictions, bucket_by_item, ks)
        coverage = evaluate_buckets.evaluate_bucketed_coverage(
            predictions,
            bucket_by_item,
            ks,
            candidate_items=bucket_by_item,
        )
        captures.append({
            bucket: {
                "num_cases": hr[bucket]["num_cases"],
                "hr": hr[bucket]["hr"],
                "coverage": coverage[bucket]["coverage"],
            }
            for bucket in evaluate_buckets.BUCKET_NAMES
        })
        return hr

    evaluate_buckets.evaluate_bucketed_hr = capture_bucket_metrics
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *script_args]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        evaluate_buckets.evaluate_bucketed_hr = original_hr
        sys.argv = original_argv
        sys.path[:] = original_path
        os.chdir(original_cwd)
    return merge_bucket_results(captures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = args.script.resolve()
    if not script.is_file():
        raise FileNotFoundError(script)
    script_args = args.script_args
    if script_args[:1] == ["--"]:
        script_args = script_args[1:]
    result = run_inference(script, script_args)
    print(f"{RESULT_PREFIX}{json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()
