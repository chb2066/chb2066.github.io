---
title: Emotion labels for Korean traditional patterns
summary: Nine unrelated methods all stalled in the same narrow band on image-only accuracy. Why I stopped pushing on the image and invested in the text side instead, and the fusion model that came out of it.
context: AiRLab · ETRI-funded
period: 2026.06 – 2026.08
role: Experiment design, training, analysis
stack: PyTorch, DINOv3, klue-roberta
tags: [Vision-Language, Knowledge Distillation, Multi-label]
date: 2026-08-26
---

## Setup

Predict which emotional adjectives describe a traditional pattern image — *abundant*, *mysterious*, *classical*. Each image carries exactly 5 labels from a 22-word vocabulary, so precision equals recall and the metric is a set-agreement rate.

```
top-5 F1 = |predicted ∩ ground truth| / 5
```

Always predicting exactly five makes this a ranking problem, not a thresholding one — that matters later. Each record also has a free-form Korean `description` written by an annotator and nine structured metadata fields. The brief required using the image together with its accompanying information, with the image influencing the prediction. Prior team result: 0.62 (FG-CLIP2). Target: 0.80.

## Where the signal is

Each channel measured alone, same split.

| Channel | Method | F1@5 |
|---|---|---|
| **Description** | klue/roberta-large, fine-tuned | **0.784** |
| Metadata (9 fields) | multi-hot → linear probe | 0.581 |
| Image | ConvNeXtV2-large, fine-tuned | 0.589 |

Text is worth about 20 points more than the image. Metadata — including the `meaning` field I expected to help — is worth the same as the pixels and no more.

## The image ceiling

A single backbone stalling proves nothing, so I attacked 0.59 from as many unrelated directions as I could.

| Approach | Best variant | F1@5 |
|---|---|---|
| Frozen features + linear probe | FG-CLIP2 / SigLIP2 / DINOv3 | 0.510 |
| VLM fine-tuning | Qwen2.5-VL-7B, LoRA | 0.557 |
| ViT backbone | EVA-02-large, end-to-end | 0.571 |
| CNN backbone | ConvNeXtV2-large + drop_path/mixup | 0.589 |
| Vision ensemble | three fine-tuned models | 0.592 |
| Captioning → text classifier | KoBART captioner | 0.586 |
| Cross-modal alignment | image → generic text embedding | 0.583 |
| Knowledge distillation | feature align + frozen teacher head | 0.590 |
| **Soft prompting** | image → 16 tokens → frozen text encoder | **0.592** |

Two more attempts made things *worse*, and both were informative. Strong augmentation (RandAugment, color jitter) dropped to 0.578 — saturation and edge orientation are plausibly causal for these labels, so standard augmentation was corrupting the target. Per-class logit offsets fitted on validation with inner 5-fold CV hurt every model by 0.001–0.008, meaning the ranking already matched label frequency. Class-balanced sampling hurt for the same reason: rebalancing targets macro-recall and works against top-*k*.

### Soft prompting

The sharpest measurement. The image is converted into input tokens for the text model, so it passes through **the same frozen encoder and head** that scores 0.78 on real descriptions. Only the translator is trained.

```
image ─▶ 16 soft tokens ─▶ [ frozen text encoder ] ─▶ [ frozen head ] ─▶ 22 logits
```

It reached 0.592. The auxiliary loss aligning the translated tokens to the real description embedding bottomed out at cosine distance ~0.66 and would not go lower — the image-reconstructed embedding points in a different direction from the text describing the same object.

At that point the image-only routes were exhausted — plain backbones, a VLM, and several attempts to pull the image representation toward the text side. None of them moved. What the sequence settled is that the image alone has a clear limit on this task, while the text reaches far higher, because the description simply carries more of what the labels depend on than the pixels do.

So the question stopped being how to extract more from the image and became how to train the text side harder. With a dataset this small, fine-tuning a BERT-family encoder was the realistic way to get there, and getting there was the point — the brief asked for a visible number, not a diagnosis.

## Fusion

The goal changed from beating 0.59 to satisfying the brief: a model in which the image measurably contributes. The first attempt ran attention over DINOv3's 256 patch tokens and collapsed — logits went nearly uniform. Cut to one pooled vector per modality.

```
image ─▶ DINOv3 (frozen) ──────▶ proj ─▶ 1 token ─┐
                                                  ├─▶ self-attn ─▶ linear ─▶ 22
text  ─▶ Korean encoder (tuned) ▶ proj ─▶ 1 token ─┘
```

The pattern images carry little information individually and look alike across the set. Give a decoder many queries under that condition and they search a space where the answers barely differ, so they interfere rather than specialise — the ML-Decoder-style arrangement was working against the data it had. Few queries should behave better, and among the ways to use very few, attention over one token per modality is the smallest arrangement that still forms the output from both distributions at once.

Concat 0.661 → self-attention 0.677 under identical settings, text encoder frozen in both. Unfreezing the text encoder moved the same architecture to ~0.78, which made **text fine-tuning the one decisive lever** in the project.

| Image encoder treatment | F1@5 |
|---|---|
| Frozen, off-the-shelf | 0.7767 |
| Frozen, domain-adapted | 0.7780 |
| Unfrozen, same LR as text | 0.7744 |
| Unfrozen, low LR | 0.7758 |

Four conditions inside 0.4 points. Opening the image backbone buys nothing, so DINOv3 stayed frozen.

### A bug

Late in the project I found that what I called "description-only" text input was in fact description plus all nine metadata fields — a comment in the training script was about which *source* the text came from, and I read it as which *field*. I added an explicit flag and retrained everything. Individual scores moved 0.003–0.005, consistent with the channel measurements above.

## Results

### Is the image used?

| Image input | F1@5 | Δ |
|---|---|---|
| **Real features** | **0.7785** | — |
| Zeroed | 0.7646 | −0.0139 |
| Shuffled across samples | 0.7709 | −0.0076 |

Run on a single fusion model, before the description-only fix and before the final combination was settled; not repeated on the final ensemble. What it establishes is that a trained fusion model reads the image at inference — the property the design had to satisfy — measured on the same architecture and recipe every ensemble member uses.

### Final system

Five text encoders, each fused with the same frozen DINOv3 features, then ensembled.

| Text encoder | F1@5 |
|---|---|
| kobigbird-bert-base | 0.7861 |
| klue/roberta-large | 0.7794 |
| xlm-roberta-large | 0.7753 |
| bert-base-multilingual-cased | 0.7753 |
| kcbert-large | 0.7682 |

All 63 subsets of six candidates, searched with Dirichlet-weighted random search: four encoders 0.7991, all six 0.7996, and the best five (klue + xlmr + kcbert + mbert + kobigbird) **0.8000**. Swapping one architecturally similar encoder for the structurally distinct kobigbird decided whether the target was met. A fully uniform learning-rate schedule reached 0.7969, so the per-model tuning was necessary rather than convenient.

### Errors

- No sample missed all five labels; 77.1% got at least four right.
- Best: *abundant* 0.909, *simple* 0.893, *cute* 0.881. Worst: *mysterious* 0.656, *luxurious* 0.660, *modern* 0.691.
- Difficulty tracks abstractness, not rarity — one of the rarest labels, *humorous*, scores 0.824, while the far more common *elegant* scores poorly.
- When uncertain, the model falls back to the two highest-frequency labels.
- The five ensemble members agree at Jaccard 0.74–0.79, differing at ambiguous boundaries rather than making unrelated predictions.

Reading the predictions, part of what the model has learned is not individual labels but label *sets* — which five tend to travel together. Inside a set it recovers even rare labels; a label that the inferred set does not imply is under-predicted no matter how common it is. That places the remaining error on the label definitions rather than on the model: the vocabulary has groupings the annotation never made explicit.

---

All numbers come from one fixed validation split with seed 42 and reproduce within ±0.2 points under bf16 non-determinism. Dataset composition and per-record details are omitted as project material.
