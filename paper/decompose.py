"""Attack strength vs per-target vulnerability profile (contribution 1).

ASR(m,t) = mu + alpha_m + beta_t + gamma_mt   (percentage points, any-turn)
  alpha_m  : attack strength (method mean over the target panel, minus mu)
  beta_t   : target susceptibility
  gamma_mt : interaction -- what single-best-attack reporting assumes is 0

Inference
  * gamma CIs: request-level bootstrap within each cell (k* ~ Binomial(n, k/n)).
  * Interaction test: binomial GLM (logit) with method + target main effects;
    its residual deviance is the LRT against the saturated (interaction) model,
    df = (M-1)(T-1). Zero cells are fine here (no separation in the additive
    model). p-value from chi-square AND from a parametric bootstrap, since
    counts are small.

Usage: python decompose.py [judge ...]     (default: qwen7b llamaguard qwen72b)
Reads analysis/counts.csv (from build_tables.py), writes analysis/decomp_<judge>.md
"""
import csv
import itertools
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from build_tables import METHODS, METHOD_NAMES, TARGETS

RNG = np.random.default_rng(0)
B = 2000


def decompose(A):
    mu = A.mean()
    alpha = A.mean(axis=1) - mu
    beta = A.mean(axis=0) - mu
    gamma = A - mu - alpha[:, None] - beta[None, :]
    return mu, alpha, beta, gamma


def additive_deviance(k, n):
    M, T = k.shape
    df = pd.DataFrame([(m, t, k[m, t], n[m, t]) for m in range(M) for t in range(T)],
                      columns=["m", "t", "k", "n"])
    X = pd.get_dummies(df[["m", "t"]].astype(str), drop_first=True).astype(float)
    X = sm.add_constant(X)
    y = np.column_stack([df["k"], df["n"] - df["k"]])
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return fit.deviance, fit.fittedvalues.to_numpy().reshape(M, T)


def run(judge, counts):
    tnames = [name for _, name in TARGETS]
    k = np.zeros((len(METHODS), len(TARGETS)))
    n = np.zeros_like(k)
    for i, m in enumerate(METHODS):
        for j, (t, _) in enumerate(TARGETS):
            r = counts[(m, t, judge)]
            if r["complete"] != "True":
                return f"# {judge}: skipped -- ({m}, {t}) not fully judged yet\n"
            k[i, j], n[i, j] = int(r["k_any"]), int(r["n"])
    A = 100 * k / n
    mu, alpha, beta, gamma = decompose(A)

    boot = np.empty((B,) + A.shape)
    for b in range(B):
        boot[b] = decompose(100 * RNG.binomial(n.astype(int), k / n) / n)[3]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    dev, p_fit = additive_deviance(k, n)
    dof = (len(METHODS) - 1) * (len(TARGETS) - 1)
    p_chi = stats.chi2.sf(dev, dof)
    sims = [additive_deviance(RNG.binomial(n.astype(int), p_fit), n)[0] for _ in range(1000)]
    p_boot = (1 + sum(s >= dev for s in sims)) / (1 + len(sims))

    out = [f"# Decomposition — judge: {judge} (any-turn, pp)\n",
           f"Grand mean mu = {mu:.2f} pp over {len(METHODS)} attacks x {len(TARGETS)} targets.\n",
           "## Attack strength vs generality\n",
           "| Attack | mean ASR | alpha (strength) | SD across targets | min | max | targets at 0% | targets where best |",
           "|---|---|---|---|---|---|---|---|"]
    best = A == A.max(axis=0, keepdims=True)
    for i, m in enumerate(METHODS):
        out.append(f"| {METHOD_NAMES[m]} | {A[i].mean():.2f} | {alpha[i]:+.2f} | {A[i].std(ddof=1):.2f} | "
                   f"{A[i].min():.2f} | {A[i].max():.2f} | {(k[i] == 0).sum()} | {best[i].sum()} |")
    out += ["", "## Target susceptibility (beta)\n", "| Target | mean ASR | beta |", "|---|---|---|"]
    for j, name in enumerate(tnames):
        out.append(f"| {name} | {A[:, j].mean():.2f} | {beta[j]:+.2f} |")

    out += ["", "## Interaction gamma (pp) with 95% bootstrap CI — * = CI excludes 0\n",
            "| Attack | " + " | ".join(tnames) + " |", "|---|" + "---|" * len(tnames)]
    for i, m in enumerate(METHODS):
        cells = []
        for j in range(len(tnames)):
            star = "*" if lo[i, j] > 0 or hi[i, j] < 0 else ""
            cells.append(f"{gamma[i, j]:+.2f}{star} [{lo[i, j]:+.1f},{hi[i, j]:+.1f}]")
        out.append(f"| {METHOD_NAMES[m]} | " + " | ".join(cells) + " |")
    nsig = int(((lo > 0) | (hi < 0)).sum())
    out += ["", f"{nsig} of {gamma.size} gamma cells have a 95% CI excluding 0. "
                "Note each row and column of gamma sums to zero by construction.\n",
            "## Interaction test (logit scale)\n",
            f"Additive model residual deviance G^2 = {dev:.1f} on {dof} df; "
            f"chi-square p = {p_chi:.2e}; parametric-bootstrap p = {p_boot:.4f} (1000 sims).\n",
            "## Ranking inversions\n", "| Pair | A>B on | B>A on |", "|---|---|---|"]
    for a, b in itertools.combinations(range(len(METHODS)), 2):
        gt, lt = int((A[a] > A[b]).sum()), int((A[a] < A[b]).sum())
        flag = " (inverts)" if gt and lt else ""
        out.append(f"| {METHOD_NAMES[METHODS[a]]} vs {METHOD_NAMES[METHODS[b]]}{flag} | {gt} | {lt} |")
    return "\n".join(out) + "\n"


def main():
    judges = sys.argv[1:] or ["qwen7b", "llamaguard", "qwen72b"]
    counts = {(r["method"], r["target"], r["judge"]): r for r in csv.DictReader(open("analysis/counts.csv"))}
    for j in judges:
        md = run(j, counts)
        open(f"analysis/decomp_{j}.md", "w", encoding="utf-8").write(md)
        print(md)


if __name__ == "__main__":
    main()
