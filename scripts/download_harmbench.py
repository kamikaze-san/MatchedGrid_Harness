"""Download HarmBench behaviors from HuggingFace and write to data/harmbench.jsonl.

The standard split has 400 behaviors with stable per-row indices. We use the row
index as the stable request_id ('hb_0000', 'hb_0001', ...), which means the
resumption logic in runner.py works correctly across reruns.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/harmbench.jsonl")
    args = parser.parse_args()

    from datasets import load_dataset

    # The 'standard' split is the canonical HarmBench behavior set.
    ds = load_dataset("walledai/HarmBench", "standard", split="train")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for i, row in enumerate(ds):
            rec = {
                "id": f"hb_{i:04d}",
                "behavior": row.get("prompt") or row.get("behavior") or row.get("Behavior"),
                "category": row.get("category") or row.get("Category") or "unknown",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} HarmBench behaviors to {out_path}")


if __name__ == "__main__":
    main()
