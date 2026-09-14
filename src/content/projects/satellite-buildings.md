---
title: Building segmentation in satellite imagery
summary: Three architecturally unrelated models all failed at inference in exactly the same way. The bug was not in any of them — it was in how train and test resolutions were reconciled.
context: Self-directed study on a past competition dataset
period: 2026.01 – 2026.02
role: Everything — analysis, training, inference, error analysis
stack: PyTorch, UAGLNet, DINOv2, Prithvi
tags: [Segmentation, Remote Sensing, Data-Centric]
date: 2026-02-28
---

## Setup

Segment building footprints in satellite images. Two numbers define the problem before any modelling starts.

| | Train | Test |
|---|---|---|
| Resolution | 1024 × 1024 | **224 × 224** |
| Count | 7,140 | **60,640** |

Train and test are at different resolutions, and the test set is 8.5× larger than the train set. Both facts end up mattering more than the choice of architecture.

Visual inspection added two more constraints. Buildings sit in mountainous, heavily wooded terrain and are small enough that some are hard to separate from background by eye. And a portion of the training masks are simply wrong.

Given a small, partially mislabelled training set against a huge test set, the obvious first move was to lean on a large pretrained model rather than fit the training data hard.

## Everything failed the same way

| Approach | Train IoU | Test |
|---|---|---|
| DINOv2 frozen + segmentation head | normal | **0.03 – 0.07** |
| Prithvi frozen + segmentation head | normal | **0.03 – 0.07** |
| UAGLNet, task-specific, fine-tuned | normal | below baseline |

Two foundation-model pipelines and one task-specific building-extraction network — different pretraining, different architecture families, different heads. Training looked healthy in all three. Inference collapsed in all three, to the same narrow band.

That pattern is the finding. A model can be wrong. Three unrelated models being wrong *identically* points at something they share, and what they shared was the data path.

## The shared mistake

Listing what each run did to resolution made it obvious.

| Model | Train input | Test input |
|---|---|---|
| DINOv2 | resize → 518 | resize → 518, then resize → 224 |
| Prithvi | resize → 518 | resize → 518, then resize → 224 |
| UAGLNet | resize → 512 | resize → 512, then resize → 224 |

Every run reconciled the 1024/224 gap by **resizing**, and the test path resized twice. For classification that is survivable. For segmentation it is not: the prediction is per-pixel, and each resize resamples the very grid the prediction is defined on. The model was being asked to label pixels that no longer corresponded to the ones it trained on.

The fix removes resizing from both sides.

- **Train** — crop 512 × 512 tiles instead of resizing the 1024 image down.
- **Test** — pad to the working size instead of resizing 224 up and back.

Same UAGLNet, same weights, same schedule. **0.03 – 0.07 → 0.65.**

Nothing about the model changed. The entire gap was the resolution handling.

## Closing the scale gap

0.65 was stable but visibly wrong in one direction: large buildings were caught reliably, small ones were missed. Training on 512 tiles while testing on 224 images means the model learned objects at roughly twice the apparent scale it would meet at inference.

So I matched the training patch size to the test image size — 224 × 224 crops, stride 200, giving 5 × 5 = 25 overlapping patches per source image.

This is a trade, not a free win. The pretrained checkpoint was built at 512, so 224 inputs give up some of that transfer. What it buys is that inference needs no resolution adjustment at all — the model sees at test time exactly the geometry it trained on. The second effect was larger.

Changing the input size meant editing the model definition and the dataset input size, since the released code assumed 512. Batch size went 14 → 56 and learning rate 1e-3 → 4e-3 to recover throughput, increased only as far as segmentation quality held.

**0.79 at threshold 0.35.**

## Then the data, and mostly it didn't work

Train IoU reached 0.8896 while the test score sat near 0.80 — close enough that training failures looked like a usable proxy for test failures. Some images stayed wrong for the whole run, so I went after them.

I logged per-patch IoU every epoch and accumulated a failure list across runs, then used it to filter the next run. What came of it:

- **Curriculum by failure count** — warm up on never-failed patches, then upweight the ambiguous ones. No gain.
- **Drop every failing patch** — score fell to 0.797. Removing hard examples removed the signal with them.
- **Drop the worst 27 images** — the intersection of "model fails here" and "labels look wrong here". Roughly neutral.

The interesting part is why the intersection was small. I built two lists independently — images with bad labels, and images the model kept failing on — and they **did not line up**. Mislabelled images were often learned fine, and some cleanly labelled ones stayed hard. Label error and model difficulty are different axes, and filtering on either one alone mostly deletes useful data.

What did help was targeted rather than subtractive. Error visualisation showed specific failure modes — shadowed regions in particular — so I added shadow augmentation and random resized crop against those, and ensembled the shadow-robust model with the base model at weighted average.

## What I take from it

The resolution bug cost the most time and taught the most. The signal that something structural was wrong was available early and for free: **three unrelated models failing to the same narrow band**. I read that as three separate model problems for longer than I should have. A single model underperforming says little; unrelated models failing identically says the fault is in what they share.

The pruning experiments are the other half. It is tempting to treat "the model keeps getting this wrong" as "this label is bad," and to delete accordingly. Measured separately, the two sets barely overlapped, and deleting on either signal alone made things worse.

---

Code: [satellite-building-segmentation](https://github.com/chb2066/satellite-building-segmentation) — UAGLNet adapted to the competition data format, patch-based dataset, and the failure-log cleaning loop.
