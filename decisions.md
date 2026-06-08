# DINO Project — Decision Log

This file documents every significant decision made during the project, with rationale.
It is intended to feed directly into the **Cadre de reproduction** and **Discussion critique** sections of the final report.

---

## 1. Paper Selection

**Decision:** DINO — *Emerging Properties in Self-Supervised Vision Transformers*, Caron et al., ICCV 2021 (arXiv:2104.14294)

**Rationale:**
- Strong industry relevance: the self-supervised pretraining paradigm (no labels needed) directly addresses annotation cost, a key bottleneck in real-world ML pipelines. Meta's DINOv2 (2023) is a direct descendant and is widely used in production systems.
- The paper's central claim is visually compelling and easy to evaluate: self-attention maps spontaneously segment objects without any supervision. This gives us a qualitative result (attention map visualizations) alongside quantitative ones (k-NN accuracy), which makes for a richer report.
- The training framework is architecturally simple (~15 lines of PyTorch pseudocode in Algorithm 1), making a rigorous reduced reproduction feasible on Colab.
- Rich ablation space: the paper's own Table 7 tests momentum encoder, multi-crop, loss type, and predictor — we can directly mirror those experiments on our reduced setting and compare findings.

**Alternatives considered:**
- MAE: clean and reproducible, but the reconstruction results are less visually striking on small datasets.
- Barlow Twins: elegant, but less impactful and fewer natural ablations.
- DDPM: most topical given the generative AI moment, but training diffusion models is too compute-heavy even at reduced scale for Colab.

---

## 2. Dataset Selection

**Decision:** STL-10

**Rationale:**
- **Resolution**: STL-10 images are 96×96 vs CIFAR-10's 32×32. DINO uses a ViT that tokenises images into patches. With patch size 8 on CIFAR-10 you get only 4×4 = 16 tokens — far too few for self-attention to produce meaningful representations or readable attention maps. On STL-10 with patch size 8 you get 12×12 = 144 tokens, which is workable.
- **Unlabeled split**: STL-10 provides 100,000 unlabeled images specifically designed for self-supervised pretraining. CIFAR-10 has no such split — its training set is fully labeled, which is the wrong setup for an SSL method.
- **Evaluation split**: STL-10 provides 8,000 labeled test images for downstream k-NN evaluation, cleanly separated from the pretraining data.
- **Precedent**: STL-10 is the standard small-scale SSL benchmark; using it makes our results comparable to other reduced reproductions in the literature.
- **Ablation quality**: With CIFAR-10 the signal would be too noisy to produce interpretable ablation results. STL-10 gives a cleaner signal, which matters since ablation is 25% of the grade.

**Alternatives considered:**
- CIFAR-10: rejected due to resolution mismatch with ViT patch tokenisation and absence of a dedicated unlabeled split.
- CIFAR-100: same resolution problem as CIFAR-10.
- Tiny-ImageNet: 64×64 resolution — better than CIFAR but still limited; STL-10 preferred for its unlabeled split.

---

## 3. Architecture Selection

**Decision:** ViT-Tiny (patch size 8, ~5–6M parameters)

**Rationale:**
- The paper uses ViT-S/16 (21M params) and ViT-B/8 (85M params). Both are too heavy for reliable Colab training.
- ViT-Tiny (~5M params, 12 heads, 192 dim, 12 blocks) is the smallest standard ViT variant and can train on Colab T4 GPUs within reasonable time (~2–3 hours for 100 epochs on STL-10).
- Patch size 8 on 96×96 images gives 144 tokens — enough for attention to be meaningful. Patch size 16 would give only 36 tokens, which risks degraded attention map quality.
- We explicitly note in the report that the architecture is reduced, not that we are claiming to match the paper's performance numbers.

---

## 4. Training Scope Simplifications

**Decision:** Train for 100 epochs (vs 300 in the paper), batch size 256 (vs 1024), 2 global crops only as default (multi-crop added in ablation)

**Rationale:**
- 100 epochs on STL-10 with ViT-Tiny fits within Colab session limits (~2–3h on T4).
- Batch size 256 is the minimum the paper tested (Table 9 shows bs=128 gives ~57.9% k-NN top-1 vs 59.1% for bs=256 — small gap, acceptable).
- Starting with 2 global crops only (no local crops) as the default setting simplifies the training loop and makes multi-crop an explicit ablation variable rather than a hidden assumption.
- The learning rate is scaled linearly with batch size per the paper's rule: lr = 0.0005 × batchsize/256.

---

## 5. Evaluation Protocol

**Decision:** k-NN accuracy on STL-10 test set (frozen features, no finetuning)

**Rationale:**
- The paper evaluates with both linear probing and k-NN. k-NN is simpler to implement (no training phase, no hyperparameter tuning), more interpretable, and directly highlights the quality of the raw learned features.
- k-NN evaluation requires only one forward pass over the dataset — feasible on Colab.
- We also produce attention map visualisations on STL-10 test images as a qualitative metric.

**Baseline:**
- A ViT-Tiny trained from scratch with standard supervised cross-entropy on STL-10's 5,000 labeled training images, evaluated with k-NN on the same test set. This provides a direct apples-to-apples comparison: same architecture, same evaluation protocol, supervised vs self-supervised.
- Random initialisation k-NN accuracy as a lower bound sanity check.

---

## 6. Ablation Plan

**Decision:** Three ablations mirroring the paper's Table 7, adapted to our reduced setting.

| Ablation | What changes | Hypothesis |
|---|---|---|
| A1 — No momentum encoder | Teacher = direct copy of student at each step | Model collapses (k-NN ≈ random) — validates that EMA teacher is load-bearing |
| A2 — No multi-crop | Remove local views (2 global only → no local-to-global) | k-NN accuracy drops ~3–5% — validates multi-crop contribution |
| A3 — Teacher temperature sweep | τ_t ∈ {0.02, 0.04, 0.07, 0.1} | Too low → over-sharpening / instability; too high → under-sharpening / collapse |

**Rationale for these three:**
- A1 tests the most critical component identified in the paper (Table 7 row 2: removing momentum → 0.1% accuracy). If we replicate collapse, it strongly validates our reproduction.
- A2 tests the second most impactful component and is easy to implement (just remove the local crop augmentation).
- A3 is an extension not explicitly in Table 7 and adds originality to our ablation section, which the grader rewards.

---

## 7. Report Scope Notes

- We will clearly state in the **Cadre de reproduction** section: (1) what is reproduced faithfully, (2) what is simplified, and (3) what is evaluated differently.
- We do not claim to match the paper's ImageNet numbers. Our goal is to demonstrate that the core mechanism works on a reduced setting and that the ablations qualitatively reproduce the paper's findings.
- Honest reporting of failures or underperformance is explicitly valued by the grader ("un résultat mauvais sera bien évalué si l'analyse est critique et honnête").

---

*Last updated: 2026-05-19*
