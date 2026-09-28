"""Judge comparison on the main grid (canonical cells only).

  1. Per-turn agreement: every (request, turn) scored by all three judges
     (errored turns and STRATA turns > 2 excluded). Binary kappa at tau=8,
     Pearson r on raw scores, and each judge's positive rate on the same turns.
  2. Stability of the decomposition across judges, on the LOGIT scale
     (levels differ by ~60pp between judges, so pp comparisons are meaningless):
     corr(alpha) over attacks, corr(gamma) over the 40 cells, gamma sign agreement.
  3. Qwen-7B vs Qwen-72B per target: Kendall tau between the attack rankings,
     and whether the top-1 attack agrees.

Usage: python judges.py   -> analysis/judges.md   (needs analysis/counts.csv)
"""
import csv
import itertools
import json
import os

import numpy as np
from scipy import stats

from build_tables import JUDGES, K_CAP, METHOD_NAMES, METHODS, TARGETS, TAU, canonical_cells, load_jsonl

JN = {"qwen7b": "Qwen-7B", "qwen72b": "Qwen-72B", "llamaguard": "Llama-Guard-3"}


def per_turn_scores():
    rows = []
    for m in METHODS:
        for t, _ in TARGETS:
            for d in canonical_cells(m, t):
                cfg = json.load(open(os.path.join(d, "config.json")))
                cap = K_CAP.get(cfg["method"].split("_")[0])
                ok = {(r["request_id"], x["turn"])
                      for r in load_jsonl(os.path.join(d, "trajectories.jsonl")) for x in r["trajectory"]
                      if not str(x.get("target_response", "")).startswith("[target_error")
                      and (cap is None or x["turn"] <= cap)}
                sc = {}
                for j, suf in JUDGES.items():
                    p = os.path.join(d, f"rejudge_perturn_{suf}.jsonl")
                    sc[j] = {(x["request_id"], x["turn"]): (x.get("score") or 0)
                             for x in load_jsonl(p)} if os.path.exists(p) else {}
                for key in ok:
                    if all(key in sc[j] for j in JUDGES):
                        rows.append({j: sc[j][key] for j in JUDGES})
    return rows


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def logit_decomp(counts, judge):
    k = np.array([[int(counts[(m, t, judge)]["k_any"]) for t, _ in TARGETS] for m in METHODS], float)
    n = np.array([[int(counts[(m, t, judge)]["n"]) for t, _ in TARGETS] for m in METHODS], float)
    L = np.log((k + 0.5) / (n - k + 0.5))  # Haldane-Anscombe correction for zero cells
    mu = L.mean()
    alpha = L.mean(axis=1) - mu
    gamma = L - mu - alpha[:, None] - (L.mean(axis=0) - mu)[None, :]
    return alpha, gamma, 100 * k / n


def main():
    counts = {(r["method"], r["target"], r["judge"]): r for r in csv.DictReader(open("analysis/counts.csv"))}
    complete = all(counts[(m, t, j)]["complete"] == "True" for m in METHODS for t, _ in TARGETS for j in JUDGES)
    out = ["# Judge comparison (main grid, any-turn)\n"]
    if not complete:
        out.append("**WARNING: not every (method, target) cell is fully judged by all three judges -- numbers are provisional.**\n")

    rows = per_turn_scores()
    out += [f"## 1. Per-turn agreement ({len(rows)} turns scored by all three judges)\n",
            "| Judge pair | kappa (binary, tau=8) | Pearson r (raw scores) |", "|---|---|---|"]
    for a, b in itertools.combinations(JUDGES, 2):
        xa = [r[a] >= TAU for r in rows]
        xb = [r[b] >= TAU for r in rows]
        pr = stats.pearsonr([r[a] for r in rows], [r[b] for r in rows])[0]
        out.append(f"| {JN[a]} vs {JN[b]} | {kappa(xa, xb):.3f} | {pr:.3f} |")
    out += ["", "| Judge | positive rate on these turns |", "|---|---|"]
    for j in JUDGES:
        out.append(f"| {JN[j]} | {100 * np.mean([r[j] >= TAU for r in rows]):.2f}% |")

    if complete:
        dec = {j: logit_decomp(counts, j) for j in JUDGES}
        out += ["", "## 2. Decomposition stability across judges (logit scale)\n",
                "| Judge pair | corr(alpha), 5 attacks | corr(gamma), 40 cells | gamma sign agreement |", "|---|---|---|---|"]
        for a, b in itertools.combinations(JUDGES, 2):
            ca = stats.pearsonr(dec[a][0], dec[b][0])[0]
            cg = stats.pearsonr(dec[a][1].ravel(), dec[b][1].ravel())[0]
            sg = np.mean(np.sign(dec[a][1]) == np.sign(dec[b][1]))
            out.append(f"| {JN[a]} vs {JN[b]} | {ca:+.2f} | {cg:+.2f} | {100 * sg:.0f}% |")
        out += ["", "Attack strength alpha (logit) by judge:\n", "| Attack | " + " | ".join(JN[j] for j in JUDGES) + " |",
                "|---|" + "---|" * len(JUDGES)]
        for i, m in enumerate(METHODS):
            out.append(f"| {METHOD_NAMES[m]} | " + " | ".join(f"{dec[j][0][i]:+.2f}" for j in JUDGES) + " |")

        A7, A72 = dec["qwen7b"][2], dec["qwen72b"][2]
        out += ["", "## 3. Qwen-7B vs Qwen-72B attack ranking per target\n",
                "| Target | Kendall tau | top-1 (7B) | top-1 (72B) | agree |", "|---|---|---|---|---|"]
        for j, (_, name) in enumerate(TARGETS):
            tau_k = stats.kendalltau(A7[:, j], A72[:, j])[0]
            top = lambda A: "=".join(METHOD_NAMES[METHODS[i]] for i in np.flatnonzero(A[:, j] == A[:, j].max()))
            t7, t72 = top(A7), top(A72)
            out.append(f"| {name} | {tau_k:+.2f} | {t7} | {t72} | {'yes' if set(t7.split('=')) & set(t72.split('=')) else 'no'} |")

    md = "\n".join(out) + "\n"
    open("analysis/judges.md", "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
