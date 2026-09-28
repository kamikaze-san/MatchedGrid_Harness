"""Offline re-judge an entire results/ directory with a different judge model.

Reads trajectories.jsonl for every cell, sends (request, final_response) to a
designated judge vllm server, writes rejudge_<judge_id>.jsonl alongside.

Resumable: skips request_ids already in the rejudge file.

Usage:
    python -m src.rejudge \\
        --judge-name meta-llama/Llama-Guard-3-8B \\
        --judge-port 8003 \\
        --results-dir results \\
        --output-suffix llama_guard_3
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judge import MultiJudge
from src.utils import atomic_append_jsonl


def rejudge_cell(cell_dir: Path, judge: MultiJudge, suffix: str):
    traj_file = cell_dir / "trajectories.jsonl"
    if not traj_file.exists():
        return 0, 0
    out_file = cell_dir / f"rejudge_{suffix}.jsonl"

    # Build set of already-rejudged request_ids
    done_set = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_set.add(rec["request_id"])
                except Exception:
                    continue

    n_new, n_skip = 0, 0
    with open(traj_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            req_id = r["request_id"]
            if req_id in done_set:
                n_skip += 1
                continue
            trajectory = r["trajectory"]
            if not trajectory:
                continue
            # The judge.score() interface needs a request dict
            request = {"behavior": _recover_behavior(r, trajectory)}
            try:
                result = judge.score(request=request, trajectory=trajectory)
            except Exception as e:
                result = {"score": -1, "reasoning": f"rejudge_error: {e}", "copy_penalised": False}
            rec = {
                "request_id": req_id,
                "category": r.get("category", "unknown"),
                "cell_id": r.get("cell_id"),
                "method": r.get("method"),
                "target": r.get("target"),
                "seed": r.get("seed"),
                "original_score": r.get("score"),
                "rejudge_score": result["score"],
                "rejudge_reasoning": result.get("reasoning", ""),
                "copy_penalised": result.get("copy_penalised", False),
                "jaccard": result.get("jaccard"),
                "judge_id": judge.name,
            }
            atomic_append_jsonl(out_file, rec)
            n_new += 1
    return n_new, n_skip


def _recover_behavior(r, trajectory):
    """The trajectory record doesn't always include the original behavior text — recover it
    from the HarmBench data file via the request_id. Cached for speed."""
    global _HB_CACHE
    if "_HB_CACHE" not in globals():
        _HB_CACHE = {}
        for hb_path in (Path("data/harmbench.jsonl"), Path("data/advbench.jsonl")):
            if hb_path.exists():
                with open(hb_path) as f:
                    for line in f:
                        obj = json.loads(line)
                        _HB_CACHE[obj["id"]] = obj["behavior"]
    # Never judge against an empty request (unknown ids used to fall back to "").
    return _HB_CACHE[r["request_id"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge-name", required=True)
    p.add_argument("--judge-port", type=int, required=True)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output-suffix", required=True)
    args = p.parse_args()

    judge = MultiJudge(primary_port=args.judge_port, primary_name=args.judge_name)
    total_new = 0
    total_skip = 0
    cells = sorted(Path(args.results_dir).glob("G1_*"))
    for cell_dir in cells:
        n, s = rejudge_cell(cell_dir, judge, args.output_suffix)
        if n or s:
            print(f"[rejudge] {cell_dir.name}: new={n} skipped={s}", flush=True)
        total_new += n
        total_skip += s
    print(f"[rejudge] done. total new={total_new} skipped={total_skip}")


if __name__ == "__main__":
    main()
