"""Crescendo with k=10 turns (Russinovich et al. 2024 default).

Identical to crescendo.py but the runner's k_turns parameter is forced to 10
to match the original paper rather than truncating at 6.
"""
from src.methods.crescendo import run_episode as _run_episode


def run_episode(request, attacker_port, attacker_model, target_port, target_model,
                k_turns, seed):
    return _run_episode(request, attacker_port, attacker_model, target_port,
                        target_model, k_turns=10, seed=seed)
