---
title: DINOv3
paper: DINOv3
venue: arXiv 2025
link: https://arxiv.org/abs/2508.10104
claim: 오래 학습할수록 무너지는 dense feature를 Gram anchoring으로 붙잡으면, 얼린 백본만으로 dense task까지 커버한다.
tags: [Self-supervised, Vision Backbone]
tier: basic
date: 2025-07-02
draft: true
---

[DINOv3](https://arxiv.org/abs/2508.10104)

## Abstract

**문제**
자기지도 학습을 더 크게, 더 오래 밀면 전체 이미지 분류는 좋아지는데 dense feature(패치 단위 품질)가 무너짐
알려져 있었지만 해결되지 않았던 문제

**해결책**
패치 값 자체가 아니라 패치 간 관계 행렬(Gram matrix) 을 과거의 자기 자신에 정박시키는 정규화
고해상도 적응 단계를 추가해 얼린 백본으로 segmentation·depth까지 커버

---

## 1. Introduction

**DINOv2에서 남은 문제**
- 라벨링되지 않은 데이터에서 쓸모 있는 데이터를 얼마나 모았는지 불분명
- 보통 cosine schedule을 쓰는데 큰 이미지를 학습할 때는 이게 좋지 않음
- 학습이 진행되면서 다루는 특징의 다양성이 점차 줄어듦 → 초기 이후에는 비슷한 데이터만 처리

이런 문제는 ViT-Large보다 큰 모델을 더 오래 학습할 때 나타나고, DINOv2도 예외가 아니다.

**연구 목표**
1. 다재다능한 기본 모델을 학습
1. dense feature에서 SSL 모델의 약점을 개선
1. 모델을 얼린 채로 써도 높은 성능

목표는 결국 하나다. 단일 frozen SSL backbone을 범용 visual encoder로 쓸 수 있게 만드는 것.

---

## 3. Training at Scale Without Supervision

- **3.1 Data Preparation** - 데이터 큐레이션. DINOv2의 retrieval 방식을 이어받되 규모를 키움
- **3.2 Large-Scale Training** - teacher는 7B 파라미터 ViT

---

## 4. Gram Anchoring: A Regularization for Dense Features

### 4.1 Loss of Patch-Level Consistency Over Training

**데이터를 많이 넣고 오래 학습시키면 사진 전체를 분류하는 능력은 좋아진다. 그런데 dense하게 보는 능력이 떨어지거나 뭉개진다.**

원인은 두 목표가 서로 상충한다는 데 있다.
- 고수준 이해(전체 이미지가 무엇인가)를 잘하려면 세부를 버리고 추상화해야 함
- dense feature 품질(각 패치가 무엇인가)을 지키려면 세부를 유지해야 함

큰 모델일수록 오래 학습하면 전자로 기울고, collapse에 취약해진다.

- 특정 패치와 나머지 패치의 cosine similarity가 학습이 진행될수록 뭉개지는 것이 보인다

### 4.2 Gram Anchoring Objective

**아이디어의 출처** - 화풍 변환 연구에서 쓰이던 Gram Matrix 개념을 차용. 각 패치가 담고 있는 값을 feature로 사용한다.

**Gram Matrix가 재는 것** - 패치와 패치 사이의 유사도. 여기서 전략이 나온다.

> 패치 각각의 값이 바뀌는 것은 괜찮다. 다만 패치들 사이의 관계는 비슷하게 유지되게 한다.

**작동 구조**
- 과거의 자기 자신을 teacher로 사용 → dense 성능이 낮아지기 전의 초기 모델을 가져와 규제
- teacher와 student에서 각각 패치 간 Gram matrix를 구함
  - feature 행렬 `X` 와 그 전치 `Xᵀ` 를 곱하는 연산
  - 패치들 사이의 관계를 나타내는 행렬이 만들어짐. attention과 비슷하지만 `X·Xᵀ` 형태라 값이 일정하게, 형태가 비슷하게 유지되는 성질을 활용
- 다음 식으로 teacher의 분포를 student가 닮아가게 함
```text
L_Gram = || Gram_student − Gram_teacher ||²
```

**이 방법이 통하는 이유**
- 표현 자체는 계속 발전해도 됨
- 전역 성능은 계속 오르게 두고, 잃어가는 축(패치 간 기하 관계)만 과거 시점에 정박
- 규제 대상을 좁힌 것이 핵심

- Gram anchoring 적용 후 dense 벤치마크가 회복되는 것이 확인됨

### 4.3 Leveraging Higher-Resolution Features

기본 학습 해상도는 효율을 위해 256으로 작게 유지하되, 고해상도가 필요한 dense task를 위해 후반부에 적응 단계를 붙인다.

- Gram teacher가 student보다 높은 해상도로 이미지를 받아 더 세밀한 patch-level feature를 뽑음
- 그 feature map을 student의 patch grid 크기에 맞춰 다운샘플링한 뒤 Gram matrix를 비교

---

## 5. Post-Training

**모델 스케일과 Distillation**
- DINOv2와 마찬가지로 거대한 teacher 하나로 여러 크기의 student(ViT-S, B, L, 커스텀 S+, H+)를 동시에 distillation
- Distillation 단계에서는 Gram anchoring을 쓰지 않음 → 기존 objective만. 작은 student는 애초에 dense feature 붕괴 문제가 덜하기 때문

**5.1 Resolution Scaling**
- 고해상도 적응 단계에서 global crop을 최대 768px까지 섞어 10k iteration 정도 추가 학습
- 이 방식으로 4k 해상도 입력에서도 안정적인 local feature가 유지됨
