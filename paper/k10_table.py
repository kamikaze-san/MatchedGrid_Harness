"""Crescendo k=10 vs k=6 (crescendo_pilot, Qwen-7B judge).

Like-for-like = seed 1 at both budgets; the k=6 3-seed pool is shown where it exists (Qwen2.5-VL-32B).
Also cumulative first-success-by-turn, to show where successes land in each budget.

Usage: python k10_table.py   -> analysis/k10.md
"""
import os

import numpy as np
from scipy import stats

from build_tables import TAU, cell_outcomes, load_jsonl

T4 = [("internvl25_8b", "InternVL2.5-8B"), ("qwen25vl7b", "Qwen2.5-VL-7B"),
      ("qwen25vl_32b", "Qwen2.5-VL-32B"), ("llava_ov_7b", "LLaVA-OV-7B")]


def first_success(d):
    sc = {}
    for x in load_jsonl(os.path.join(d, "rejudge_perturn_qwen7b_perturn.jsonl")):
        if (x.get("score") or 0) >= TAU:
            sc[x["request_id"]] = min(sc.get(x["request_id"], 99), x["turn"])
    return sc


def main():
    L = ["# Crescendo k=10 vs k=6 (Qwen-7B judge)\n",
         "| Target | k=6 any (seed 1) | k=10 any (seed 1) | p (paired) | k=6 final | k=10 final | k=6 any, 3 seeds | errored turns k=10 |",
         "|---|---|---|---|---|---|---|---|"]
    curves = []
    for t, name in T4:
        d6, d10 = f"results/PILOT_{t}_crescendo_pilot_seed1", f"results/K10_{t}_crescendo_pilot_seed1"
        o6, o10 = cell_outcomes(d6, "qwen7b_perturn"), cell_outcomes(d10, "qwen7b_perturn")
        if o10 is None:
            L.append(f"| {name} | | k=10 not fully judged | | | | | |")
            continue
        a6 = {k: v[0] for k, v in o6.items()}
        a10 = {k: v[0] for k, v in o10.items()}
        common = a6.keys() & a10.keys()
        x = sum(a10[i] and not a6[i] for i in common)
        y = sum(a6[i] and not a10[i] for i in common)
        p = stats.binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
        seeds = [f"results/PILOT_{t}_crescendo_pilot_seed{s}" for s in (1, 2, 3)]
        pool = [cell_outcomes(d, "qwen7b_perturn") for d in seeds if os.path.isdir(d)]
        pooled = (100 * sum(v[0] for o in pool for v in o.values()) / sum(len(o) for o in pool)
                  if len(pool) == 3 and all(o is not None for o in pool) else None)
        rs = load_jsonl(os.path.join(d10, "trajectories.jsonl"))
        err = 100 * sum(str(z["target_response"]).startswith("[target_error") for r in rs for z in r["trajectory"]) / \
            sum(len(r["trajectory"]) for r in rs)
        pc = lambda o, i: 100 * sum(v[i] for v in o.values()) / len(o)
        L.append(f"| {name} | {pc(o6, 0):.2f} (n={len(o6)}) | {pc(o10, 0):.2f} (n={len(o10)}) | {p:.2f} | "
                 f"{pc(o6, 1):.2f} | {pc(o10, 1):.2f} | {f'{pooled:.2f}' if pooled is not None else '--'} | {err:.1f}% |")
        f6, f10 = first_success(d6), first_success(d10)
        curves.append((name, [100 * sum(v <= k for v in f6.values()) / len(o6) for k in range(1, 7)],
                       [100 * sum(v <= k for v in f10.values()) / len(o10) for k in range(1, 11)]))
    L += ["", "## Cumulative any-turn success by turn (%)\n", "| Target | budget | " + " | ".join(f"t{k}" for k in range(1, 11)) + " |",
          "|---|---|" + "---|" * 10]
    for name, c6, c10 in curves:
        L.append(f"| {name} | k=6 | " + " | ".join(f"{v:.1f}" for v in c6) + " |" + " |" * 4)
        L.append(f"|  | k=10 | " + " | ".join(f"{v:.1f}" for v in c10) + " |")
    L.append("\nThe attacker is told its budget each turn, so a k=10 episode is a different escalation plan, "
             "not a k=6 episode with four extra turns.\n")
    md = "\n".join(L) + "\n"
    open("analysis/k10.md", "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
