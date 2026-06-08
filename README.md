# DINO — Reduced Reproduction on STL-10

Reduced but rigorous reproduction of **DINO** (Caron et al., *Emerging Properties in
Self-Supervised Vision Transformers*, ICCV 2021, [arXiv:2104.14294](https://arxiv.org/abs/2104.14294))
for the *Introduction to Deep Learning* validation project (Albert School, B&D 3rd year).

We reproduce DINO's core self-distillation mechanism with a ViT-Tiny/8 backbone on STL-10,
evaluate frozen features with k-NN against a supervised baseline, visualise the emergent
self-attention maps, and run three ablations mirroring the paper's Table 7.

## Contents

```
dino_stl10.ipynb     # main reproducible notebook (run top to bottom on Colab)
decisions.md         # decision log (paper/dataset/arch/ablation rationale)
DINO.pdf             # the reproduced paper
report/              # LaTeX report + compiled PDF
figures/             # generated figures (loss curve, attention maps)
results/             # generated JSON result files
checkpoints/         # saved model checkpoints (git-ignored)
requirements.txt
```

## How to run (Google Colab)

1. Open `dino_stl10.ipynb` in Colab; set **Runtime → Change runtime type → GPU (T4)**.
2. Leave `FAST_MODE = True` (cell *Run configuration*) for the deadline run. It subsamples
   the unlabeled split and shortens training so the **whole pipeline finishes in ~2–3 h**.
   Set `FAST_MODE = False` for the paper-faithful settings (~8–12 h, multi-session).
3. **Run all cells, top to bottom.** Order of stages:

   | Stage | Cell | FAST_MODE wall-clock (T4) | Produces |
   |-------|------|---------------------------|----------|
   | Setup + config + data | top | ~5 min (STL-10 download) | — |
   | Main DINO pretraining | §5 | ~30–45 min | `checkpoints/`, `figures/loss_curve.png` |
   | k-NN eval (DINO + random) | §7 | ~2 min | `dino_knn`, `random_knn` |
   | Supervised baseline | §7b | ~10 min | `results/main_results.json` |
   | Attention maps | §8 | ~1 min | `figures/attention_*.png` |
   | Ablations A1, A2, A3 | §9 | ~1.5 h | `ablation_results` |
   | Summary | end | <1 min | `results/all_results.json` |

4. After the run, **download `results/all_results.json` and the `figures/` PNGs** and paste
   the numbers into `report/` where marked `\TODO{...}`.

## What to paste back into the report

- `results/main_results.json` → `dino_knn`, `random_knn`, `supervised_knn`, `supervised_test_cls`
- `results/all_results.json` → the `ablations` dict (A1/A2/A3 k-NN accuracies)
- `figures/loss_curve.png`, `figures/attention_*.png`

## Reproducibility notes

- Global seed = 42; the unlabeled subsample is deterministic (`PRETRAIN_IDX`).
- All design choices and their rationale are documented in `decisions.md`.
- FAST_MODE is an **explicit, honest reduction** of the paper setting (documented in the
  report's *Cadre de reproduction*); the goal is to demonstrate the mechanism, not to match
  the paper's ImageNet numbers.
