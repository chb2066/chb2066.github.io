---
title: Paragraph-level AI text detection as multiple-instance learning
summary: Between rounds the unit of scoring moved from the document to the paragraph, while the rules still allowed paragraphs of one document to see each other. That combination is a bag, so I rebuilt it as MIL.
context: 2025 SW-Centered University Digital Competition, AI Division
period: 2025.07 – 2025.08
role: Implementation
stack: PyTorch, Hugging Face Transformers, DDP
tags: [NLP, Multiple-Instance Learning, Calibration]
date: 2025-08-12
---

## The task changed shape

The campus round asked one question per text: human or AI. Winning it sent us to the national round, where the question looked similar and was not.

| | Campus round | National round |
|---|---|---|
| Scored unit | whole text | **each paragraph** |
| Output | one label | probability per paragraph |
| Cross-reference | n/a | **paragraphs of the same document may reference each other** |

The third row is the design lever. Evaluation rows carry a `title`, and rows sharing a title belong to one article — and using them together is explicitly permitted.

That matters because a single paragraph is a weak piece of evidence. Three sentences of neutral prose look the same from either author. The document around it is far more informative, but the score is collected per paragraph, so you cannot simply classify the document and copy the answer down.

## Why MIL is the right shape

A group of instances that must be judged individually, where the group carries signal the individual does not, is a **bag**. That is the setting multiple-instance learning was built for, so the task was reframed rather than merely tuned: split each document on newlines, tokenise the paragraphs separately, and feed the whole bag to the model.

The open question is where aggregation sits relative to prediction. The two orders lead to genuinely different models, so both were implemented.

### EPA — Embed → Predict → Aggregate

```
paragraphs ─▶ encoder ─▶ per-paragraph logits ─▶ top-k ─▶ self-attention ─▶ document
```

Keeps a paragraph-level classifier and trains both levels at once:

```
loss = doc_loss + λ · paragraph_loss
```

Paragraphs are ranked by their own logits, the top-k are kept, and multi-head self-attention combines them. Because a paragraph head exists and is supervised, the per-paragraph output is a first-class prediction rather than a by-product.

### EAP — Embed → Aggregate → Predict

```
paragraphs ─▶ encoder ─▶ top-k by embedding norm ─▶ pool ─▶ one document prediction
```

No paragraph head, one document-level loss. Importance comes from the embedding norm rather than a trained score, and pooling is selectable — mean, max, attention, weighted attention — with `top_k_ratio` 0.7 by default.

The trade is visible in the diagrams. EPA spends parameters on judging each paragraph and can explain which ones drove the document; EAP commits earlier to a single document representation and asks less of the paragraph level.

## Two details that mattered

**The loss was built to match the metric.** Scoring is AUC and the labels are imbalanced, so cross-entropy is the wrong target twice over. The models optimise focal loss combined with a differentiable AUC surrogate, weighted 0.3, with the focal parameters exposed (`gamma` 3.0, `alpha` 0.75). Optimising a relaxation of the metric instead of a proxy for it is the cheapest alignment available.

**The output is a probability, so it has to be calibrated.** Submissions are probabilities in [0, 1], not rankings, so a model that orders correctly but is systematically overconfident is still wrong. Expected calibration error is tracked alongside AUC during training, and the final checkpoint is temperature-scaled on held-out data before it is used.

Training runs under `DistributedDataParallel` with mixed precision; prediction runs on a single GPU and emits calibrated probability with a confidence score beside it.

## What I take from it

The useful move was noticing the task had changed shape, not just changed granularity. "Score each paragraph" and "paragraphs of a document may see each other" are two facts that only pay off when read together — separately, the first suggests a paragraph classifier and the second suggests a document classifier, and both are worse than treating the document as a bag.

Building both aggregation orders rather than picking one on intuition was the right call too. Predict-then-aggregate and aggregate-then-predict are equally defensible on paper, and the difference between them is not something I could have reasoned out in advance.

---

Code: [AI_generated_text_detection](https://github.com/chb2066/AI_generated_text_detection) — both models, the combined loss, and the calibration pass.
