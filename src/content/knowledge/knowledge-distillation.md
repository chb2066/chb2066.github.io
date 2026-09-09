---
title: Knowledge Distillation
summary: What actually transfers from teacher to student — logits, features, or relations.
tags: [Knowledge Distillation]
date: 2026-08-15
draft: false
---

## Definition

Transferring what a large model (teacher) has learned to a smaller model, or to a model in a different modality (student).

## What gets transferred

- **Logit KD** — match the teacher's output distribution as a soft target. When the teacher's output is nearly identical to the labels, this adds no information the hard targets did not already carry.
- **Feature KD** — align intermediate representations. Often a bigger win than imitating outputs.
- **Relational KD** — transfer the structure between samples rather than per-sample values.

> **To write** — this is the format for a knowledge base entry. Rewrite it the way I actually understand it.

## What I saw

In the traditional-pattern project, logit KD gained nothing and only feature alignment helped. Numbers are in the [project note](/projects/traditional-patterns/).
