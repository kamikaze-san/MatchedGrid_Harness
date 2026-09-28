# Method Ranking Inverts Across Targets — code

Code for *Method Ranking Inverts Across Targets: A Matched-Grid Evaluation of Jailbreak Attacks on Open-Weight
Vision–Language Models*. It contains the evaluation harness (five attacks, vLLM-served targets, per-turn
judging), the configurations of every run reported in the paper, and the scripts that produce every table and
figure from the released trajectories.

**Data:** the trajectories and per-turn judge scores are on the Hugging Face Hub at `<hf-dataset-id>`
(gated; research use only).

> **Content warning.** The harness generates attacks from harmful behaviour prompts (HarmBench, AdvBench) and
> records model responses to them. Use it only for safety research.

## Reproduce the paper's tables and figures (no GPU needed)

```bash
pip install -r requirements.txt
python paper/hf_to_results.py --source <hf-dataset-id>   # rebuild results/ from the released dataset
python paper/build_tables.py      # Table 1 and per-cell ASR  -> analysis/
python paper/decompose.py         # decomposition and interaction test (Section 5.2)
python paper/judges.py            # judge agreement (Table 3, Section 5.3)
python paper/appendix_latex.py    # every appendix table      -> analysis/appendix/
python paper/make_figures.py      # Table 1 LaTeX and figures
```

Run these from the repository root. On the released data they reproduce the paper's numbers exactly.

## Layout

| Path | What |
|---|---|
| `src/runner.py` | runs one cell: loads its config, drives the attack against the target, judges each episode |
| `src/methods/` | the attacks: FigStep, PAIR, Crescendo, STRATA (`*_pilot.py`, `crescendo_k10.py` are the variants used for the added targets and the 10-turn run; `crescendo.py` is the earlier Crescendo prompt used by the attacker controls) |
| `src/modality/` | deterministic image and PDF rendering for FigStep and STRATA |
| `src/judge.py`, `src/rejudge_per_turn.py` | judge prompt and offline per-turn scoring |
| `sbatch/run_cell.sbatch` | Slurm job for one cell: starts the target and attacker vLLM servers, then the runner |
| `sbatch/rejudge_per_turn.sbatch` | Slurm job that scores every turn of saved trajectories with one judge |
| `configs/cells/` | the 102 cell configs reported in the paper; `configs/reported_cells.json` groups them by experiment |
| `data/` | HarmBench (200) and AdvBench (200) behaviours with the request ids used throughout |
| `scripts/` | environment setup, model download, HarmBench download |
| `paper/` | analysis scripts for every table and figure |

## Running the harness

Requirements: NVIDIA GPUs (we used A100-80GB; the 26B/32B targets and the Qwen2.5-72B judge need a GPU of this
size), Slurm, conda.

```bash
bash scripts/setup_env.sh            # conda env "strata": Python 3.11, vLLM 0.7.3, transformers 4.49
bash scripts/download_models.sh      # all models used in the paper into $HF_HOME
```

Run one cell (edit the partition/QoS/GPU lines at the top of the sbatch files for your cluster):

```bash
sbatch --export=ALL,CELL_CFG=configs/cells/G1_internvl25_8b_figstep_seed1.json,TARGET_MAX_LEN=16384,MM_LIMIT=2,STRATA_IMG_WINDOW=2 sbatch/run_cell.sbatch
```

Settings used in the paper (Appendix O):

| Variable | Paper value | Meaning |
|---|---|---|
| `TARGET_MAX_LEN` | 16384 (32768 for LLaVA-OneVision) | target context window |
| `MM_LIMIT` | 2 | images per prompt the target server accepts |
| `STRATA_IMG_WINDOW` | 2 | STRATA keeps the two most recent rendered pages in context |
| `STRATA_MAX_TURNS` | 2 (optional) | STRATA is analysed on turns 1–2; the analysis applies this cap, so setting it only saves compute |

Optional: `MODULES` (modules to load), `CONDA_SH` (path to `conda.sh`), `CONDA_ENV`, `HF_HOME`,
`TARGET_GPU_UTIL` / `ATTACKER_GPU_UTIL` (vLLM memory fractions; with one GPU per job, e.g. 0.40 each).
A cell resumes from its `done.txt` if interrupted.

Score saved trajectories per turn with a judge:

```bash
sbatch --export=ALL,JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct,JUDGE_SUFFIX=qwen7b_perturn,CELL_GLOB='*' sbatch/rejudge_per_turn.sbatch
sbatch --export=ALL,JUDGE_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ,JUDGE_SUFFIX=qwen72b_perturn,CELL_GLOB='*' sbatch/rejudge_per_turn.sbatch
sbatch --export=ALL,JUDGE_MODEL=meta-llama/Llama-Guard-3-8B,JUDGE_SUFFIX=llamaguard3_perturn,CELL_GLOB='*' sbatch/rejudge_per_turn.sbatch
```

Then run the `paper/` scripts as above on your own `results/`.

## Citation

```bibtex
@inproceedings{thamatam2026ranking,
  title     = {Method Ranking Inverts Across Targets: A Matched-Grid Evaluation of Jailbreak Attacks on Open-Weight Vision--Language Models},
  author    = {Thamatam, Venkat Aniruddh and Majhi, Vasudev and Nagda, Aaryan and Challa, Jagat Sesh and Mandal, Murari and Sinha, Yash and Kumar, Dhruv},
  booktitle = {TODO: venue},
  year      = {TODO}
}
```
