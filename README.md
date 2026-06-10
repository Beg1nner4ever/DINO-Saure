# DINO — Reduced Reproduction on STL-10

Reduced but rigorous reproduction of **DINO** (Caron et al., *Emerging Properties in
Self-Supervised Vision Transformers*, ICCV 2021, [arXiv:2104.14294](https://arxiv.org/abs/2104.14294))
for the *Introduction to Deep Learning* validation project (Albert School, B&D 3rd year).

We reproduce DINO's core self-distillation mechanism with a ViT-Tiny/8 backbone on STL-10,
evaluate frozen features with k-NN against random-init and supervised baselines, visualise the
emergent self-attention maps, and run three ablations inspired by the paper's Table 7
(momentum encoder, multi-crop, teacher temperature).

> **The notebook already ships with executed outputs**, and the report
> (`report/rapport_dino.pdf`) is fully filled with the results below — you do **not** need to
> re-run anything to read the work. Re-running is only needed to reproduce from scratch.

## Headline results (k-NN top-1 on STL-10 test, frozen features)

| | k-NN top-1 |
|---|---|
| Supervised baseline (same backbone) | 52.6% |
| **DINO (self-supervised, adaptive)** | **37.2%** |
| Random init | 27.4% |
| Ablation A1 — no momentum (collapse) | 19.6% |
| Ablation A2 — no multi-crop | 31.6% |
| Ablation A3 — teacher temperature sweep | 32.4 / 31.5 / 30.9% |

Raw values are in `results/*.json`, figures in `figures/`. The run was done locally on
**Apple Silicon (MPS)**.

## Contents

```
dino_stl10.ipynb     # main notebook, WITH executed outputs (run top to bottom)
decisions.md         # decision log (paper/dataset/arch/ablation rationale)
DINO.pdf             # the reproduced paper
Projet_ASDL.pdf      # the assignment brief
report/              # LaTeX source + compiled PDF (rapport_dino.pdf)
figures/             # generated figures (architecture, ablations, attention, curves)
results/             # JSON result files (the data behind the figures/tables)
checkpoints/         # saved model checkpoints (git-ignored)
requirements.txt
```

## How to run

The pipeline detects the device automatically (CUDA → Apple MPS → CPU).

1. Open `dino_stl10.ipynb` (Colab: **Runtime → GPU (T4)**, or locally on Apple Silicon / a GPU).
2. Keep `FAST_MODE = True` (config cell) — the reduced, deadline-feasible setting. `FAST_MODE = False`
   gives the paper-faithful settings (much longer, multi-session).
3. **Run all cells, top to bottom.** Pretraining uses the adaptive protocol (§5b); the fixed-epoch
   §5 cell is skipped by default (`RUN_FIXED_PRETRAIN = False`).

| Stage | Section | Produces |
|-------|---------|----------|
| Setup + config + data | §1–4 | downloads STL-10 |
| **Adaptive DINO pretraining** | §5b | `checkpoints/dino_adaptive_best.pt`, `figures/adaptive_curve.png` |
| k-NN eval (DINO + random) | §7 | `results/main_results.json` |
| Supervised baseline | §7b | (added to `main_results.json`) |
| Attention maps | §8 | `figures/attention_*.png` |
| Ablations A1 / A2 / A3 | §9 | `results/all_results.json` |
| Summary | end | prints the results table above |

**Compute note.** The full FAST_MODE pipeline is multi-hour. On Colab's free T4 we hit GPU
out-of-memory at the default batch and the session idle-timeout, so the reported run was done
locally on Apple MPS (see the report's *Contraintes d'exécution* section). The notebook is now
**T4-safe**: `BATCH_SIZE` is device-aware (128 on CUDA, 256 on MPS) to avoid the OOM.

## Reproducibility notes

- Global seed = 42; the unlabeled subsample is deterministic (`PRETRAIN_IDX`).
- k-NN evaluation uses a probe set carved from a **held-out slice of the labeled train split**, so
  early-stopping / model selection never touches the test set.
- The ablations are run **at matched training length** (20 epochs); the multi-crop effect is
  measured against the 20-epoch full-config baseline (A3 τ_t=0.04), not the longer-trained main run
  (see the report's ablation discussion).
- All design choices and their rationale are documented in `decisions.md`.
- FAST_MODE is an **explicit, honest reduction** of the paper setting; the goal is to demonstrate
  the mechanism, not to match the paper's ImageNet numbers.
