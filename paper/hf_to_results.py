"""Rebuild the results/ directory layout from the released dataset, so the paper/ scripts can run on it.

  python paper/hf_to_results.py --source <hf-dataset-id>      # download from the Hugging Face Hub
  python paper/hf_to_results.py --source path/to/dataset_dir   # a local copy with data/*.parquet

Writes results/<cell_id>/{config.json, trajectories.jsonl, rejudge_perturn_<judge>.jsonl}, which is what
build_tables.py, decompose.py, robustness.py, judges.py and appendix_latex.py read.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

SUFFIX = {"qwen2.5-7b": "qwen7b_perturn", "qwen2.5-72b": "qwen72b_perturn", "llama-guard-3": "llamaguard3_perturn"}
JUDGE_ID = {"qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct", "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct-AWQ",
            "llama-guard-3": "meta-llama/Llama-Guard-3-8B"}
CONFIGS = ["main_grid", "attacker_controls", "advbench", "crescendo_k10"]


def load(source):
    if os.path.isdir(source):
        import pyarrow.parquet as pq
        for c in CONFIGS:
            for f in sorted(glob.glob(os.path.join(source, "data", f"{c}*.parquet"))):
                yield from pq.read_table(f).to_pylist()
    else:
        from datasets import load_dataset
        for c in CONFIGS:
            yield from load_dataset(source, c, split="train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    cells = defaultdict(list)
    for r in load(args.source):
        cells[r["cell_id"]].append(r)
    for cell_id, recs in cells.items():
        d = os.path.join(args.out, cell_id)
        os.makedirs(d, exist_ok=True)
        r0 = recs[0]
        json.dump({"cell_id": cell_id, "target": r0["target_model"], "attacker": r0["attacker_model"],
                   "method": r0["method"], "seed": r0["seed"], "n_requests": len(recs),
                   "k_turns": r0["turn_budget"]}, open(os.path.join(d, "config.json"), "w"), indent=2)
        judged = defaultdict(list)
        with open(os.path.join(d, "trajectories.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                traj = [{"turn": t["turn"], "user_text": t["user_text"], "modality_text": t["rendered_text"],
                         "attacker_plan": t["attacker_plan"], "attacker_raw": t["attacker_output"],
                         # errored turns are released without the server's error text; restore the marker
                         # the analysis uses to exclude them
                         "target_response": "[target_error]" if t["errored"] else t["target_response"]}
                        for t in r["turns"]]
                # run-time judge score = primary judge on the last analysed turn (they agree exactly; Appendix O)
                used = [t for t in r["turns"] if t["analysed"] and (t["scores"] or {}).get("qwen2.5-7b")]
                final = used[-1]["scores"]["qwen2.5-7b"]["score"] if used else None
                f.write(json.dumps({"request_id": r["request_id"], "category": r["category"], "cell_id": cell_id,
                                    "seed": r["seed"], "method": r["method"], "target": r["target_model"],
                                    "score": final, "trajectory": traj}, ensure_ascii=False) + "\n")
                for t in r["turns"]:
                    for j, sc in (t["scores"] or {}).items():
                        if sc is not None:
                            judged[j].append({"request_id": r["request_id"], "turn": t["turn"], "cell_id": cell_id,
                                              "method": r["method"], "target": r["target_model"], "seed": r["seed"],
                                              "score": sc["score"], "reasoning": sc["reasoning"],
                                              "copy_penalised": t["copy_penalised"], "jaccard": t["bigram_jaccard"],
                                              "judge_id": JUDGE_ID[j]})
        for j, rows in judged.items():
            with open(os.path.join(d, f"rejudge_perturn_{SUFFIX[j]}.jsonl"), "w", encoding="utf-8") as f:
                f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    print(f"wrote {len(cells)} cells to {args.out}/")


if __name__ == "__main__":
    main()
