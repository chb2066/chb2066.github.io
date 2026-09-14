---
title: Building a training set for scenes you cannot film
summary: The system had to work in three environments. We could only film in one and a half of them. What closed the gap was not more footage — it was more kinds of footage, and a label definition we could actually apply twice.
context: Hanbat National University AiRLab
period: 2026.05 – 2026.07
role: Data construction, training, error analysis
stack: PyTorch, object detection
tags: [Object Detection, Data-Centric, Video]
date: 2026-07-31
---

## Setup

Detect situations that need attention in a subway station, split across two detectors.

| Detector | Classes | Why grouped |
|---|---|---|
| People | fallen, crouching | Both are body-posture judgements on a person |
| Fire | flame, smoke | Both are fire evidence, and smoke often appears without visible flame |

Two models rather than one four-class model, because the two groups are different kinds of visual evidence. A posture is a person in an unusual configuration; smoke has no shape to speak of. Asking one detector to hold both classes of concept together buys nothing when the classes never co-occur in a way that needs joint reasoning.

No public dataset covered these targets in this setting, so the training set had to be built.

## The constraint that shaped everything

The brief asked for one thing that turned out to drive the whole project: the system had to work **inside a subway station, inside a lab, and inside the research institute**.

We could film in an auditorium, and in the institute. That is it.

So two of the three required environments were places we could not record in, and one of them — the subway station — was the primary deployment target. The question stopped being "can we detect a fallen person" and became **"can a model trained where we can film work where we cannot."**

## First pass, and why it wasn't enough

We filmed, labelled, trained. Two problems surfaced together.

**Not enough data.** Expected, and the easy one to state.

**The label criteria would not survive contact with a second annotator — or with the same annotator on a second day.** Two boundaries in particular:

- *Crouching* — where does it start? A person tying a shoelace, sitting low on their heels, leaning down to pick something up. Some are the target and some are not, and the first pass had no line written down.
- *Smoke* — where does it end? Smoke has no edge. Thin haze at the frame margin either is or is not part of the instance, and the answer changed from clip to clip.

The second problem is the worse one, because it is invisible in the loss. An inconsistent boundary does not look like a bug; it looks like a hard example the model has not learned yet.

## Second pass

Three changes, and only one of them is about volume.

**Pin down the labels first.** Write the boundary for crouching and for smoke, then apply it uniformly — including relabelling what already existed. Everything downstream depends on the target meaning the same thing in every clip.

**Bring in outside footage and label it ourselves.** AI Hub has relevant video. Taking it under *our* label definition rather than its original one kept the set consistent instead of stapling two annotation conventions together.

**Widen the environments, not just the count.** Filming moved to the auditorium *and outside the auditorium*, plus several additional locations. The point was not more frames — it was more backgrounds, lighting conditions, and camera geometries per target class.

## The result that mattered

After the second pass, the detectors worked well **in lab environments that appeared in neither the training nor the validation set**.

That is the part worth keeping. Not that performance improved — that it improved *in an environment the model had never seen*, which is precisely what the brief demanded and precisely what we could not film. Generalisation to the unreachable environments came from environment variety in what we could reach, not from volume, and not from anything about the architecture.

The label work was the enabler rather than the cause. With an inconsistent boundary, adding environments adds noise; the model cannot tell "new background" from "annotator changed their mind." The definitions had to be fixed before diversity could pay.

## Stationary light sources

The fire detector had a specific and persistent false positive: lamps, signage, and other fixed bright sources read as flame. Colour and texture put them close to fire, and a single frame carries nothing that separates them.

Motion does. Fire moves — it flickers, spreads, and changes shape frame to frame. A ceiling light does not change at all. Comparing across frames instead of judging each one alone separates the two cleanly, and it works regardless of how convincingly a given lamp resembles flame in a still image.

The general form is worth writing down: when a false positive is indistinguishable in the representation you are using, the fix is often a different representation rather than a better model on the same one. Here the missing axis was time, and it was free — the input was already video.

## What I take from it

The instinct when a model underperforms on self-collected data is to collect more of it. Both real levers here were something else: making the label mean one thing, and covering more *kinds* of scene rather than more scenes. More footage of the same auditorium would not have produced detection in a lab.

And the light-source fix is the same lesson as the fire/posture split at the top — match the representation to what actually distinguishes the classes. A still frame cannot separate flame from a lamp, and no amount of training on still frames will change that.

---

Numbers are omitted as project material. Written from the process rather than the results.
