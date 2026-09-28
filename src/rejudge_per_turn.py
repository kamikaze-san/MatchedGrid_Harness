"""Offline PER-TURN re-judge over already-saved trajectories.jsonl.

Unlike rejudge.py (which scores only the final turn, same as the inline judge),
this scores every turn of every trajectory by slicing the saved trajectory and
calling MultiJudge.score() on each prefix. MultiJudge.score() already only reads
trajectory[-1], so trajectory[:i] correctly reproduces "the judge as of turn i" —
same prompt, same copy-penalty logic (scoped to the payload seen by that turn),
zero changes needed to judge.py. No generation happens here — purely a re-read
of data already on disk.

Output: rejudge_perturn_<suffix>.jsonl, one line per (request_id, turn).
Resumable: skips (request_id, turn) pairs already present.

Usage (e.g. Crescendo cells only, using the primary Qwen-7B judge):
    python -m src.rejudge_per_turn \\
        --judge-name Qwen/Qwen2.5-7B-Instruct \\
        --judge-port 8002 \\
        --results-dir results \\
        --output-suffix qwen7b_perturn \\
        --cell-filter crescendo
"""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judge import MultiJudge
from src.utils import atomic_append_jsonl


def rejudge_cell_per_turn(cell_dir: Path, judge: MultiJudge, suffix: str):
    traj_file = cell_dir / "trajectories.jsonl"
    if not traj_file.exists():
        return 0, 0
    out_file = cell_dir / f"rejudge_perturn_{suffix}.jsonl"

    done_set = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_set.add((rec["request_id"], rec["turn"]))
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
            trajectory = r.get("trajectory") or []
            if not trajectory:
                continue
            behavior = _recover_behavior(r, trajectory)
            request = {"behavior": behavior}
            n_turns_total = len(trajectory)

            for i in range(1, n_turns_total + 1):
                if (req_id, i) in done_set:
                    n_skip += 1
                    continue
                prefix = trajectory[:i]
                try:
                    result = judge.score(request=request, trajectory=prefix)
                except Exception as e:
                    result = {"score": -1, "reasoning": f"rejudge_error: {e}", "copy_penalised": False}
                rec = {
                    "request_id": req_id,
                    "turn": i,
                    "n_turns_total": n_turns_total,
                    "cell_id": r.get("cell_id"),
                    "method": r.get("method"),
                    "target": r.get("target"),
                    "seed": r.get("seed"),
                    "original_final_score": r.get("score"),
                    "score": result["score"],
                    "reasoning": result.get("reasoning", ""),
                    "copy_penalised": result.get("copy_penalised", False),
                    "jaccard": result.get("jaccard"),
                    "judge_id": judge.name,
                }
                atomic_append_jsonl(out_file, rec)
                n_new += 1
    return n_new, n_skip


def _recover_behavior(r, trajectory):
    global _HB_CACHE
    if "_HB_CACHE" not in globals():
        _HB_CACHE = {}
        for hb_path in (Path("data/harmbench.jsonl"), Path("data/advbench.jsonl")):
            if hb_path.exists():
                with open(hb_path) as f:
                    for line in f:
                        obj = json.loads(line)
                        _HB_CACHE[obj["id"]] = obj["behavior"]
    # Never judge against an empty request: an unknown id used to fall back to "",
    # which silently scored every AdvBench turn without its harmful behaviour.
    return _HB_CACHE[r["request_id"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge-name", required=True)
    p.add_argument("--judge-port", type=int, required=True)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output-suffix", required=True)
    p.add_argument("--cell-glob", default="G1_*",
                    help="Glob pattern for cell directories (default: G1_*)")
    p.add_argument("--cell-filter", default=None,
                    help="Only process cell dirs whose name contains this substring "
                         "(e.g. 'crescendo' to scope to Crescendo cells only)")
    args = p.parse_args()

    judge = MultiJudge(primary_port=args.judge_port, primary_name=args.judge_name)
    total_new = 0
    total_skip = 0
    cells = sorted(Path(args.results_dir).glob(args.cell_glob))
    if args.cell_filter:
        cells = [c for c in cells if args.cell_filter in c.name]
    print(f"[rejudge_per_turn] {len(cells)} cell(s) selected", flush=True)
    for cell_dir in cells:
        n, s = rejudge_cell_per_turn(cell_dir, judge, args.output_suffix)
        if n or s:
            print(f"[rejudge_per_turn] {cell_dir.name}: new={n} skipped={s}", flush=True)
        total_new += n
        total_skip += s
    print(f"[rejudge_per_turn] done. total new={total_new} skipped={total_skip}")


if __name__ == "__main__":
    main()
