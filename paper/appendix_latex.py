"""Appendix sections as LaTeX, generated from the final data (any-turn, tau=8 unless stated).

Writes analysis/appendix/<name>.tex, one file per appendix section, each with the
\\section heading and the \\label the main text references:

  full_grid        app:full_grid          per-cell ASR + Wilson 95% CI, 8 targets x 5 attacks x 3 judges
  anyturn          app:anyturn            final-turn vs any-turn, every cell, every judge
  gamma            app:gamma              gamma matrix (+ alpha, beta) under each judge, bootstrap CIs
  thresholds       app:thresholds         interaction test across tau, per judge
  categories       app:categories         per-category ASR (4 targets) + interaction test per category/judge
  advbench         app:advbench           AdvBench FigStep vs PAIR, final and any-turn, + 2x4 interaction test
  attackers        app:attacker_controls  Llama / Qwen-7B / Qwen-72B attacker controls
  top1_bootstrap   app:top1_bootstrap     P(top-1 flip) for all 28 target pairs under each judge
  mcnemar          app:mcnemar            anchor reversal under each judge + primary-judge top-1 vs others
  judge_details    app:judge_details      alpha/gamma agreement across judges; 7B vs 72B ranking per target
  copy_penalty     app:copy_penalty       copy-penalty firing per STRATA cell, worst-case bound
  crescendo_k10    app:crescendo_k10      k=10 vs k=6 and cumulative success by turn

Usage: python appendix_latex.py
"""
import csv
import itertools
import json
import os

import numpy as np
from scipy import stats

from build_tables import JUDGES, METHOD_NAMES, METHODS, TARGETS, TAU, canonical_cells, cell_outcomes, load_jsonl, wilson
from decompose import additive_deviance, decompose
from judges import logit_decomp
from robustness import grid, request_records

OUT = "analysis/appendix"
JN = {"qwen7b": "Qwen2.5-7B", "qwen72b": "Qwen2.5-72B", "llamaguard": "Llama-Guard-3"}
JORDER = ["qwen7b", "qwen72b", "llamaguard"]
ORIG4 = ["internvl25_8b", "qwen25vl7b", "qwen25vl_32b", "llava_ov_7b"]
NAME = dict(TARGETS)
SHORT = {"internvl25_8b": "InternVL-8B", "internvl25_26b": "InternVL-26B", "qwen25vl7b": "Qwen-VL-7B",
         "qwen25vl_32b": "Qwen-VL-32B", "llava_ov_7b": "LLaVA-OV", "phi35_vision": "Phi-3.5-V",
         "minicpm_v26": "MiniCPM-V", "pixtral_12b": "Pixtral-12B"}
MN = {m: (METHOD_NAMES[m].replace("STRATA", r"\strata{}")) for m in METHODS}
RNG = np.random.default_rng(0)
C = {(r["method"], r["target"], r["judge"]): r for r in csv.DictReader(open("analysis/counts.csv"))}
REC = {}


def recs(j):
    if j not in REC:
        REC[j] = request_records(j)
    return REC[j]


def kn(m, t, j, key="k_any"):
    r = C[(m, t, j)]
    return int(r[key]), int(r["n"])


def pct(m, t, j, key="k_any"):
    k, n = kn(m, t, j, key)
    return 100 * k / n


def fmt_p(p):
    if p >= 0.001:
        return f"{p:.3f}"
    e = int(np.floor(np.log10(p)))
    return rf"${p / 10 ** e:.1f}{{\times}}10^{{{e}}}$" if e > -300 else r"$<10^{-300}$"


def table(env, cols, header, rows, caption, label, size=r"\footnotesize", resize=False):
    # adjustbox max width: shrinks a table only if it is wider than the column/page; never enlarges it.
    # Placement: every table floats, so LaTeX keeps them in number order. Full-width tables can only go to the top of
    # a page or a float page in two-column mode ([!tp]); column tables may also go at the bottom ([!tbp]).
    # ([h] is invalid for table* and made every later table queue behind it onto half-empty float pages.)
    width = r"\textwidth" if env == "table*" else r"\columnwidth"
    place = "[!tp]" if env == "table*" else "[!tbp]"
    L = [rf"\begin{{{env}}}{place}", r"\centering", size, rf"\begin{{adjustbox}}{{max width={width}}}",
         rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header, r"\midrule"] + rows + \
        [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
         rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\end{{{env}}}"]
    return "\n".join(L)


def _inner(t):
    """Body of a table() float without its \\begin/\\end lines."""
    return "\n".join(t.split("\n")[1:-1])


def side_by_side(a, b):
    """Two small tables in one full-width float, each in a half-width minipage with its own caption and number."""
    mp = lambda t: r"\begin{minipage}[t]{0.48\textwidth}" + "\n" + _inner(t) + "\n" + r"\end{minipage}"
    return "\n".join([r"\begin{table*}[!tp]", mp(a) + r"\hfill", mp(b), r"\end{table*}"])


def stacked(a, b):
    """A small table above a wide one, in one full-width float (both keep their captions and numbers)."""
    return "\n".join([r"\begin{table*}[!tp]", r"\centering", r"\begin{minipage}{0.5\textwidth}", _inner(a), r"\end{minipage}",
                      r"\par\vspace{1.2em}", _inner(b), r"\end{table*}"])


def section(title, label, body):
    return rf"\section{{{title}}}" + "\n" + rf"\label{{{label}}}" + "\n\n" + body + "\n"


def write(name, text):
    with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8") as f:
        f.write("% Generated by appendix_latex.py from the final data -- do not edit by hand.\n" + text)
    print("wrote", name)


# ---------------------------------------------------------------- sections
def full_grid():
    rows = []
    for ti, (t, name) in enumerate(TARGETS):
        for i, m in enumerate(METHODS):
            cells = []
            for j in JORDER:
                k, n = kn(m, t, j)
                lo, hi = wilson(k, n)
                cells.append(f"{100 * k / n:.1f} [{lo:.1f}, {hi:.1f}]")
            rows.append(f"{name if i == 0 else ''} & {MN[m]} & {kn(m, t, 'qwen7b')[1]} & " + " & ".join(cells) + r" \\")
        if ti < len(TARGETS) - 1:
            rows.append(r"\midrule")
    hdr = r"\textbf{Target} & \textbf{Attack} & $n$ & " + " & ".join(rf"\textbf{{{JN[j]}}}" for j in JORDER) + r" \\"
    t = table("table*", "llrccc", hdr, rows,
              r"Per-cell ASR (\%, any-turn) with Wilson 95\% intervals under each judge. $n$ counts requests with at "
              r"least one non-errored turn, pooled over seeds; it is identical across judges.", "tab:full_grid")
    # First appendix file: float settings for the table-heavy appendix (let full-width tables stack at page tops and
    # fill float pages instead of leaving one table per half-empty page).
    params = (r"\renewcommand{\dbltopfraction}{0.9}\renewcommand{\dblfloatpagefraction}{0.8}"
              r"\setcounter{dbltopnumber}{3}" + "\n"
              r"\renewcommand{\topfraction}{0.9}\renewcommand{\floatpagefraction}{0.8}\renewcommand{\textfraction}{0.07}"
              "\n"
              # float pages: stack tables from the top instead of centring them with large gaps
              r"\makeatletter\setlength{\@fptop}{0pt}\setlength{\@dblfptop}{0pt}"
              r"\setlength{\@fpsep}{14pt}\setlength{\@dblfpsep}{14pt}"
              r"\setlength{\@fpbot}{0pt plus 1fil}\setlength{\@dblfpbot}{0pt plus 1fil}\makeatother" "\n\n")
    return params + section("Per-Cell ASR with Confidence Intervals", "app:full_grid",
                            r"Table~\ref{tab:full_grid} gives every cell of Table~\ref{tab:full_results} with its "
                            r"sample size and Wilson 95\% interval." + "\n\n" + t)


def anyturn():
    rows = []
    for ti, (t, name) in enumerate(TARGETS):
        for i, m in enumerate(METHODS):
            cells = [f"{pct(m, t, j, 'k_final'):.1f} & {pct(m, t, j):.1f}" for j in JORDER]
            rows.append(f"{name if i == 0 else ''} & {MN[m]} & " + " & ".join(cells) + r" \\")
        if ti < len(TARGETS) - 1:
            rows.append(r"\midrule")
    hdr = (r" & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{JN[j]}}}}}" for j in JORDER) + r" \\" + "\n"
           + " ".join(rf"\cmidrule(lr){{{3 + 2 * k}-{4 + 2 * k}}}" for k in range(3)) + "\n"
           + r"\textbf{Target} & \textbf{Attack} & " + " & ".join(["final & any"] * 3) + r" \\")
    t = table("table*", "llcccccc", hdr, rows,
              r"Final-turn and any-turn ASR (\%) for every cell. Final-turn scores the last non-errored turn; "
              r"any-turn counts a request as successful if any turn scores $\geq\tau$. FigStep is single-turn, so the "
              r"two coincide. PAIR stops at its first apparently compliant turn, so they differ by at most 1 point. "
              r"Under the Qwen judges, \strata{} differs by at most 2 points and Crescendo, which always runs its full "
              r"budget, by up to 4.3; under Llama-Guard-3 the multi-turn gaps are larger (up to 13 points for "
              r"\strata{} and 31 for Crescendo).", "tab:anyturn")
    return section("Final-Turn versus Any-Turn ASR", "app:anyturn",
                   r"Table~\ref{tab:anyturn} reports both metrics for every cell. The two differ only for multi-turn "
                   r"attacks, where a success at an earlier turn need not persist to the final one. We do not interpret "
                   r"these non-persistent successes causally, since later turns also carry the most direct requests."
                   + "\n\n" + t)


def gamma():
    tabs = []
    for j in JORDER:
        k = np.array([[kn(m, t, j)[0] for t, _ in TARGETS] for m in METHODS], float)
        n = np.array([[kn(m, t, j)[1] for t, _ in TARGETS] for m in METHODS], float)
        A = 100 * k / n
        mu, alpha, beta, g = decompose(A)
        rng = np.random.default_rng(0)  # fixed per judge for reproducibility
        boot = np.array([decompose(100 * rng.binomial(n.astype(int), k / n) / n)[3] for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        rows = []
        for i, m in enumerate(METHODS):
            cells = [f"{g[i, c]:+.1f}" + ("$^{*}$" if lo[i, c] > 0 or hi[i, c] < 0 else "") for c in range(len(TARGETS))]
            rows.append(f"{MN[m]} & " + " & ".join(cells) + f" & {alpha[i]:+.1f}" + r" \\")
        rows.append(r"\midrule")
        rows.append(r"$\beta_t$ & " + " & ".join(f"{b:+.1f}" for b in beta) + r" & \\")
        hdr = r"\textbf{Attack} & " + " & ".join(SHORT[t] for t, _ in TARGETS) + r" & $\alpha_m$ \\"
        tabs.append(table("table*", "l" + "c" * len(TARGETS) + "c", hdr, rows,
                          rf"Target-specific fit $\gamma_{{mt}}$ (percentage points) under {JN[j]}, with general strength "
                          rf"$\alpha_m$ and target susceptibility $\beta_t$ (grand mean $\mu = {mu:.2f}$). $^{{*}}$: 95\% "
                          r"bootstrap interval (2{,}000 within-cell resamples) excludes zero. Each row and column of "
                          r"$\gamma$ sums to zero.", f"tab:gamma_{j}"))
    return section("Decomposition under Each Judge", "app:gamma",
                   r"Tables~\ref{tab:gamma_qwen7b}--\ref{tab:gamma_llamaguard} give the full decomposition under each "
                   r"judge. Levels differ greatly between judges, so values are comparable within a table, not across "
                   r"tables." + "\n\n" + "\n\n".join(tabs))


def thresholds():
    rows = []
    for j in JORDER:
        for tau in ([6, 7, 8, 9] if j != "llamaguard" else [8]):
            k, n = grid(recs(j), tau)
            dev, _ = additive_deviance(k, n)
            p = stats.chi2.sf(dev, 28)
            rows.append(f"{JN[j]} & {tau} & {100 * k.sum() / n.sum():.2f} & {dev:.1f} & {fmt_p(p)}" + r" \\")
        if j != JORDER[-1]:
            rows.append(r"\midrule")
    t = table("table", "lcccc", r"\textbf{Judge} & $\tau$ & \textbf{Mean ASR} & $G^2$ & $p$ \\", rows,
              r"Likelihood-ratio test of the additive model (28 df) at each success threshold $\tau$. Llama-Guard-3 is "
              r"binary, so only one threshold applies. Under Qwen2.5-72B, $\tau{=}7$ and $\tau{=}8$ coincide because the "
              r"judge does not output a score of 7.", "tab:thresholds")
    return section("Robustness to the Success Threshold", "app:thresholds",
                   r"Table~\ref{tab:thresholds} repeats the interaction test of Section~\ref{sec:decomposition_results} "
                   r"at every threshold. The additive model is rejected at every threshold under every judge." + "\n\n" + t)


def categories():
    R = recs("qwen7b")
    cats = sorted({r["cat"] for v in R.values() for r in v})
    rows = []
    for ci, c in enumerate(cats):
        for i, m in enumerate(METHODS):
            cells = []
            for t in ORIG4:
                rs = [r for r in R[(m, t)] if r["cat"] == c]
                cells.append(f"{100 * np.mean([r['max_score'] >= TAU for r in rs]):.1f}" if rs else "--")
            label = c.replace("_", "/") if i == 0 else ""
            rows.append(f"{label} & {MN[m]} & " + " & ".join(cells) + r" \\")
        if ci < len(cats) - 1:
            rows.append(r"\midrule")
    t1 = table("table*", "llcccc", r"\textbf{Category} & \textbf{Attack} & " + " & ".join(NAME[t] for t in ORIG4) + r" \\",
               rows, r"ASR (\%, any-turn, primary judge) by HarmBench harm category on the four original targets.",
               "tab:categories")
    rows = []
    for c in cats:
        cells = []
        for j in JORDER:
            k, n = grid(recs(j), TAU, keep=lambda r, c=c: r["cat"] == c)
            dev, _ = additive_deviance(k, n)
            cells.append(f"{int(k.sum())} & {dev:.1f} & {fmt_p(stats.chi2.sf(dev, 28))}")
        rows.append(c.replace("_", "/") + " & " + " & ".join(cells) + r" \\")
    hdr = (r" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{\textbf{{{JN[j]}}}}}" for j in JORDER) + r" \\" + "\n"
           + " ".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(3)) + "\n"
           + r"\textbf{Category} & " + " & ".join([r"succ. & $G^2$ & $p$"] * 3) + r" \\")
    t2 = table("table*", "l" + "ccc" * 3, hdr, rows,
               r"Interaction test within each harm category over all eight targets (28 df). ``succ.'' is the number of "
               r"successful requests in the category; categories with few successes have little power, so a "
               r"non-significant result there is not evidence of additivity.", "tab:categories_test")
    return section("Harm Categories", "app:categories",
                   r"Table~\ref{tab:categories} breaks ASR down by harm category, and Table~\ref{tab:categories_test} "
                   r"repeats the interaction test within each category." + "\n\n" + t1 + "\n\n" + t2)


def advbench():
    R = recs("qwen7b")

    def inline(d):
        return {json.loads(l)["request_id"]: (json.loads(l).get("score") or 0) >= TAU
                for l in open(os.path.join(d, "trajectories.jsonl"), encoding="utf-8")}

    def mcn(a, b):
        k = a.keys() & b.keys()
        x = sum(a[i] and not b[i] for i in k)
        y = sum(b[i] and not a[i] for i in k)
        return stats.binomtest(x, x + y, 0.5).pvalue if x + y else 1.0

    rows, K = [], []
    for t in ORIG4:
        df, dp = f"results/DS2_{t}_figstep_seed1", f"results/DS2_{t}_pair_seed1"
        fa = {r: a for r, (a, _) in cell_outcomes(df, "qwen7b_perturn").items()}
        pa = {r: a for r, (a, _) in cell_outcomes(dp, "qwen7b_perturn").items()}
        ff, pf = inline(df), inline(dp)
        P = lambda o: 100 * sum(o.values()) / len(o)
        K.append([sum(fa.values()), sum(pa.values())])
        hf = 100 * np.mean([r["max_score"] >= TAU for r in R[("figstep", t)]])
        hp = 100 * np.mean([r["max_score"] >= TAU for r in R[("pair", t)]])
        rows.append(f"{NAME[t]} & {P(ff):.1f} & {P(fa):.1f} & {P(pf):.1f} & {P(pa):.1f} & {fmt_p(mcn(fa, pa))} "
                    f"& {hf:.1f} & {hp:.1f}" + r" \\")
    k = np.array(K, float).T
    dev, _ = additive_deviance(k, np.full_like(k, 200))
    hdr = (r" & \multicolumn{2}{c}{\textbf{FigStep}} & \multicolumn{2}{c}{\textbf{PAIR}} & & "
           r"\multicolumn{2}{c}{\textbf{HarmBench}} \\" + "\n"
           r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){7-8}" + "\n"
           r"\textbf{Target} & final & any & final & any & $p$ & FigStep & PAIR \\")
    t = table("table", "lccccccc", hdr, rows,
              r"AdvBench replication (200 behaviours, one seed, primary judge). Final-turn scores were assigned at run "
              r"time; on HarmBench this matches offline final-turn scoring on every one of 4{,}800 FigStep and PAIR "
              r"requests. $p$: paired McNemar test of FigStep against PAIR (any-turn). HarmBench columns: any-turn ASR "
              r"from Table~\ref{tab:full_results}.", "tab:advbench")
    return section("AdvBench Replication", "app:advbench",
                   rf"Table~\ref{{tab:advbench}} repeats the FigStep--PAIR comparison on AdvBench. On this 2-attack "
                   rf"$\times$ 4-target grid the additive model is also rejected ($G^2 = {dev:.1f}$, 3 df, "
                   rf"$p = {fmt_p(stats.chi2.sf(dev, 3)).strip('$')}$). FigStep's lead on InternVL2.5-8B and "
                   r"LLaVA-OV-7B is significant; PAIR's leads on the two Qwen targets are small and not significant."
                   + "\n\n" + t)


def attackers():
    def err(d):
        rs = load_jsonl(os.path.join(d, "trajectories.jsonl"))
        tt = sum(len(r["trajectory"]) for r in rs)
        return 100 * sum(str(x.get("target_response", "")).startswith("[target_error") for r in rs for x in r["trajectory"]) / tt

    def out(d):
        return {r: a for r, (a, _) in cell_outcomes(d, "qwen7b_perturn").items()}

    def mcn(a, b):
        k = a.keys() & b.keys()
        x = sum(a[i] and not b[i] for i in k)
        y = sum(b[i] and not a[i] for i in k)
        return fmt_p(stats.binomtest(x, x + y, 0.5).pvalue) if x + y else "--"

    P = lambda o: 100 * sum(o.values()) / len(o)
    rows = []
    for i, t in enumerate(ORIG4):
        base, fam, big = (f"results/G1_{t}_pair_seed1", f"results/FAMILY2_{t}_pair_seed1", None)
        b, f = out(base), out(fam)
        rows.append(f"{'PAIR' if i == 0 else ''} & {NAME[t]} & {P(b):.1f} & {P(f):.1f} & {mcn(f, b)} & -- & --" + r" \\")
    rows.append(r"\midrule")
    for i, t in enumerate(ORIG4):
        base, fam, big = (f"results/CRESfix_G1_{t}_crescendo_seed1", f"results/FAMILY_{t}_crescendo_seed1",
                          f"results/E3fix_{t}_crescendo72b_seed1")
        b, f, g = out(base), out(fam), out(big)
        rows.append(f"{'Crescendo' if i == 0 else ''} & {NAME[t]} & {P(b):.1f} ({err(base):.0f}) & {P(f):.1f} ({err(fam):.0f}) & "
                    f"{mcn(f, b)} & {P(g):.1f} ({err(big):.0f}) & {mcn(g, b)}" + r" \\")
    hdr = (r"\textbf{Method} & \textbf{Target} & \textbf{Qwen2.5-7B} & \textbf{Llama-3.1-8B} & $p$ & "
           r"\textbf{Qwen2.5-72B} & $p$ \\")
    t = table("table*", "llccccc", hdr, rows,
              r"Attacker controls (ASR \%, any-turn, primary judge, seed 1). Each row changes only the attacker; $p$ is a "
              r"paired McNemar test against the Qwen2.5-7B attacker. The Crescendo controls used an earlier version of "
              r"the Crescendo attacker prompt, shared by all runs within the control, so absolute values differ from "
              r"Table~\ref{tab:full_results}. Parentheses: percentage of turns cut short by the target's context "
              r"limit in these runs, which were not repaired; errored turns are excluded, and truncation can only "
              r"lower ASR. --: not run (the scale control uses Crescendo only).", "tab:attackers")
    return section("Attacker Controls", "app:attacker_controls",
                   r"Table~\ref{tab:attackers} gives the family and scale controls of Section~\ref{sec:controls}." + "\n\n" + t)


def top1_bootstrap():
    res = {}
    for j in JORDER:
        rng = np.random.default_rng(0)  # fixed per judge, so the table is reproducible regardless of call order
        by = {key: {} for key in recs(j)}
        for key, v in recs(j).items():
            for r in v:
                by[key].setdefault(r["rid"], []).append(r["max_score"] >= TAU)
        rids = sorted({r for d in by.values() for r in d})

        def mat(sample):
            A = np.zeros((len(METHODS), len(TARGETS)))
            for i, m in enumerate(METHODS):
                for c, (t, _) in enumerate(TARGETS):
                    v = [x for rid in sample for x in by[(m, t)].get(rid, [])]
                    A[i, c] = np.mean(v) if v else 0
            return A

        boots = [mat(list(rng.choice(rids, len(rids)))) for _ in range(1000)]
        for a, b in itertools.combinations(range(len(TARGETS)), 2):
            def flip(A):
                ta, tb = int(np.argmax(A[:, a])), int(np.argmax(A[:, b]))
                return ta != tb and A[ta, a] > A[tb, a] and A[tb, b] > A[ta, b]
            res[(a, b, j)] = 100 * np.mean([flip(B) for B in boots])
    rows = [f"{TARGETS[a][1]} & {TARGETS[b][1]} & " + " & ".join(f"{res[(a, b, j)]:.1f}" for j in JORDER) + r" \\"
            for a, b in itertools.combinations(range(len(TARGETS)), 2)]
    t = table("table*", "llccc", r"\multicolumn{2}{l}{\textbf{Target pair}} & " + " & ".join(rf"\textbf{{{JN[j]}}}" for j in JORDER) + r" \\",
              rows, r"Percentage of 1{,}000 bootstrap resamples (behaviours drawn with replacement) in which the two "
                    r"targets' top-ranked attacks swap: the two top attacks differ and each beats the other on its own "
                    r"target.", "tab:top1_bootstrap")
    return section("Bootstrap Stability of Top-Ranked Attacks", "app:top1_bootstrap",
                   r"Table~\ref{tab:top1_bootstrap} reports, for every pair of targets, how often their top-ranked "
                   r"attacks swap under resampling of behaviours." + "\n\n" + t)


def mcnemar():
    def pair_test(R, t, a, b):
        A = {(r["seed"], r["rid"]): r["max_score"] >= TAU for r in R[(a, t)]}
        B = {(r["seed"], r["rid"]): r["max_score"] >= TAU for r in R[(b, t)]}
        k = A.keys() & B.keys()
        x = sum(A[i] and not B[i] for i in k)
        y = sum(B[i] and not A[i] for i in k)
        return x, y, (stats.binomtest(x, x + y, 0.5).pvalue if x + y else 1.0), len(k)

    rows = []
    for j in JORDER:
        for t, (a, b) in (("internvl25_8b", ("figstep", "pair")), ("qwen25vl7b", ("pair", "figstep"))):
            x, y, p, n = pair_test(recs(j), t, a, b)
            rows.append(f"{JN[j]} & {NAME[t]} & {MN[a]} $>$ {MN[b]} & {x} & {y} & {fmt_p(p)}" + r" \\")
    t1 = table("table*", "lllccc", r"\textbf{Judge} & \textbf{Target} & \textbf{Comparison} & A only & B only & $p$ \\", rows,
               r"The FigStep--PAIR reversal of Section~\ref{sec:finding_inversion} under each judge: paired exact McNemar "
               r"test on shared behaviours. ``A only'': behaviours on which only the first attack succeeds.", "tab:mcnemar_anchor")
    R = recs("qwen7b")
    rows = []
    for ti, (t, name) in enumerate(TARGETS):
        a = [pct(m, t, "qwen7b") for m in METHODS]
        top = METHODS[int(np.argmax(a))]
        for m in METHODS:
            if m == top:
                continue
            x, y, p, n = pair_test(R, t, top, m)
            rows.append(f"{name} & {MN[top]} vs.\\ {MN[m]} & {n} & {x} & {y} & {fmt_p(p)}" + r" \\")
        if ti < len(TARGETS) - 1:
            rows.append(r"\midrule")
    t2 = table("table*", "llcccc", r"\textbf{Target} & \textbf{Top attack vs.\ other} & paired $n$ & top only & other only & $p$ \\",
               rows, r"Each target's top-ranked attack against every other attack under the primary judge (paired exact "
                     r"McNemar). Where the top attack's lead is not significant, the target's top rank is not "
                     r"statistically determined.", "tab:mcnemar_top")
    return section("Paired Significance Tests", "app:mcnemar",
                   r"Table~\ref{tab:mcnemar_anchor} tests both sides of the reversal in Section~\ref{sec:finding_inversion} "
                   r"under every judge; Table~\ref{tab:mcnemar_top} tests each target's top-ranked attack against the "
                   r"others under the primary judge." + "\n\n" + t1 + "\n\n" + t2)


def judge_details():
    dec = {j: logit_decomp(C, j) for j in JORDER}
    rows = []
    for a, b in itertools.combinations(JORDER, 2):
        ca = stats.pearsonr(dec[a][0], dec[b][0])[0]
        cg = stats.pearsonr(dec[a][1].ravel(), dec[b][1].ravel())[0]
        sg = 100 * np.mean(np.sign(dec[a][1]) == np.sign(dec[b][1]))
        rows.append(f"{JN[a]} / {JN[b]} & {ca:+.2f} & {cg:+.2f} & {sg:.0f}" + r" \\")
    t1 = table("table", "lccc", r"\textbf{Judge pair} & corr($\alpha$) & corr($\gamma$) & same sign (\%) \\", rows,
               r"Agreement of the decomposition across judges, on the logit scale (levels differ by tens of points "
               r"between judges). corr($\alpha$) is over the five attacks, corr($\gamma$) over the 40 cells.",
               "tab:judge_decomp", resize=True)
    A7 = {m: [pct(m, t, "qwen7b") for t, _ in TARGETS] for m in METHODS}
    A72 = {m: [pct(m, t, "qwen72b") for t, _ in TARGETS] for m in METHODS}
    rows = []
    for c, (t, name) in enumerate(TARGETS):
        v7 = [A7[m][c] for m in METHODS]
        v72 = [A72[m][c] for m in METHODS]
        tau = stats.kendalltau(v7, v72)[0]
        top = lambda v: "=".join(MN[METHODS[i]] for i in np.flatnonzero(np.array(v) == max(v)))
        rows.append(f"{name} & {tau:+.2f} & {top(v7)} & {top(v72)}" + r" \\")
    t2 = table("table", "lccc", r"\textbf{Target} & Kendall $\tau$ & top (7B) & top (72B) \\", rows,
               r"Agreement between the Qwen2.5-7B and Qwen2.5-72B rankings of the five attacks on each target.",
               "tab:judge_rank", resize=True)
    return section("Judge Comparison Details", "app:judge_details",
                   r"Table~\ref{tab:judge_decomp} compares the decomposition across judges; Table~\ref{tab:judge_rank} "
                   r"compares the two Qwen judges' attack rankings target by target." + "\n\n" + side_by_side(t1, t2))


def copy_penalty():
    rows = []
    for m in ("strata_image", "strata_pdf"):
        for t, name in TARGETS:
            fired, touched, nreq = 0, set(), 0
            for d in canonical_cells(m, t):
                s = json.load(open(os.path.join(d, "config.json")))["seed"]
                rs = load_jsonl(os.path.join(d, "rejudge_perturn_qwen7b_perturn.jsonl"))
                nreq += len({x["request_id"] for x in rs})
                for x in rs:
                    if x.get("copy_penalised") and x["turn"] <= 2:
                        fired += 1
                        touched.add((s, x["request_id"]))
            rows.append(f"{MN[m]} & {name} & {fired} & {len(touched)} / {nreq} & +{100 * len(touched) / nreq:.1f}" + r" \\")
        if m == "strata_image":
            rows.append(r"\midrule")
    t = table("table*", "llccc", r"\textbf{Attack} & \textbf{Target} & \textbf{Turns penalised} & \textbf{Requests affected} & "
              r"\textbf{Worst case (pp)} \\", rows,
              r"Copy-penalty activity on \strata{} (analysed turns 1--2). FigStep, PAIR and Crescendo never trigger it. "
              r"Penalised turns are not judged, so their unpenalised scores are unknown; the last column is the ASR "
              r"increase if every affected request had succeeded.", "tab:copy_penalty")
    return section("Copy-Penalty Activity", "app:copy_penalty",
                   r"Table~\ref{tab:copy_penalty} lists how often the copy penalty fired on each \strata{} cell." + "\n\n" + t)


def crescendo_k10():
    T4 = [(t, NAME[t]) for t in ORIG4]
    rows, curves = [], []
    for t, name in T4:
        d6, d10 = f"results/PILOT_{t}_crescendo_pilot_seed1", f"results/K10_{t}_crescendo_pilot_seed1"
        o6, o10 = cell_outcomes(d6, "qwen7b_perturn"), cell_outcomes(d10, "qwen7b_perturn")
        a6 = {k: v[0] for k, v in o6.items()}
        a10 = {k: v[0] for k, v in o10.items()}
        k = a6.keys() & a10.keys()
        x = sum(a10[i] and not a6[i] for i in k)
        y = sum(a6[i] and not a10[i] for i in k)
        p = stats.binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
        P = lambda o, i: 100 * sum(v[i] for v in o.values()) / len(o)
        rows.append(f"{name} & {P(o6, 1):.1f} & {P(o6, 0):.1f} & {P(o10, 1):.1f} & {P(o10, 0):.1f} & {fmt_p(p)}" + r" \\")

        def first(d):
            f = {}
            for z in load_jsonl(os.path.join(d, "rejudge_perturn_qwen7b_perturn.jsonl")):
                if (z.get("score") or 0) >= TAU:
                    f[z["request_id"]] = min(f.get(z["request_id"], 99), z["turn"])
            return f
        f6, f10 = first(d6), first(d10)
        curves.append((name, [100 * sum(v <= q for v in f6.values()) / len(o6) for q in range(1, 7)],
                       [100 * sum(v <= q for v in f10.values()) / len(o10) for q in range(1, 11)]))
    hdr = (r" & \multicolumn{2}{c}{\textbf{$k{=}6$}} & \multicolumn{2}{c}{\textbf{$k{=}10$}} & \\" + "\n"
           r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}" + "\n"
           r"\textbf{Target} & final & any & final & any & $p$ \\")
    t1 = table("table", "lccccc", hdr, rows,
               r"Crescendo at $k{=}6$ and $k{=}10$ turns (seed 1, primary judge, ASR \%). $p$: paired McNemar test of "
               r"any-turn success, $k{=}10$ against $k{=}6$.", "tab:k10", resize=True)
    rows = []
    for name, c6, c10 in curves:
        rows.append(f"{name} & $k{{=}}6$ & " + " & ".join(f"{v:.1f}" for v in c6) + " & & & &" + r" \\")
        rows.append(f" & $k{{=}}10$ & " + " & ".join(f"{v:.1f}" for v in c10) + r" \\")
    t2 = table("table*", "ll" + "c" * 10, r"\textbf{Target} & \textbf{Budget} & " + " & ".join(f"$t{{=}}{q}$" for q in range(1, 11)) + r" \\",
               rows, r"Cumulative Crescendo ASR (\%) by turn: the share of requests that have succeeded at or before "
                     r"each turn. Successes concentrate in the last turns of each budget.", "tab:k10_turns")
    return section("Crescendo with $k{=}10$ Turns", "app:crescendo_k10",
                   r"Table~\ref{tab:k10} compares Crescendo at the original paper's ten-turn budget with our default of "
                   r"six. The attacker is told its budget each turn, so a ten-turn episode follows a different "
                   r"escalation plan rather than extending a six-turn one. Table~\ref{tab:k10_turns} shows when "
                   r"successes occur." + "\n\n" + stacked(t1, t2))


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("full_grid", full_grid), ("anyturn", anyturn), ("gamma", gamma), ("thresholds", thresholds),
                     ("categories", categories), ("advbench", advbench), ("attackers", attackers),
                     ("top1_bootstrap", top1_bootstrap), ("mcnemar", mcnemar), ("judge_details", judge_details),
                     ("copy_penalty", copy_penalty), ("crescendo_k10", crescendo_k10)]:
        write(name, fn())


if __name__ == "__main__":
    main()
