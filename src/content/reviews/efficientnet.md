---
title: EfficientNet
paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
venue: ICML 2019
link: https://arxiv.org/abs/1905.11946
claim: depth·width·resolution을 따로 키우는 대신 고정된 비율로 함께 키우는 compound scaling이 같은 연산량에서 더 좋다.
take: 배율이 작을 때와 클 때 최적 비율이 다를 수 있지 않냐는 의문이 남지만, 경향성이 거의 유사해서 단순한 제약만으로도 좋은 방향을 준다.
tags: [Backbone, Model Scaling]
tier: main
date: 2025-04-02
draft: false
---

#### abstract

**개요:**
cnn은 고정 파라미터에서 개발되고 자원을 많이 쓸 수 있으면 더 좋아졌지만 너무 늘리면 안좋아졌다.
(ex: Resnet에서 depth를 과하게 늘리자 성능이 감소했고 이것을 skip connection 등으로 해결한 사례)
따라서 기존에는 depth, width, resolution 중 하나만 늘렸다.
**질의:**
네트워크를 더 넓게 더 깊게 하는 것 + 자원의 밸런스(d, w ,resolution)가 어떨 때 최고의 성능을 낼까?

#### introduction

**제안:**
- new scaling method
  - grid seach를 통해서 depth, width, resolution 등을 파라미터로 사용해서 실험을 진행해봄.
    - α= depth 계수(레이어 수)
    - β = width 계수(채널 수)
    - γ= 입력 이미지의 해상도
→ 예시: depth는 건물의 층 수, width는 각 층의 면적을 의미한다고 보면 편함. width의 의미로 동일한 층 내에서 channel 수를 늘림에 따라 얻을 수 있는 특징 수가 늘어난다.)
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

#### Compound Model Scaling

**baseline network**
α* β^2 *γ^2=2 에 수렴하게 제한함. (O(r² × w² × d), 빅오 사용)
(model scaling을 2배씩 하기 위함.)

#### EfficientNet Architecture

> *[그림 자리 — Notion 원본에서 옮겨야 함]*

- conv1, conv6 는 channels를 1배, 6배한 것을 의미함.
  1. channels n배(1 or 6)
  1. convolution
  1. 압축하는 구조
    1. “채널 확장, conv, 채널 압축 구조” 의 구성
**STEP 1:(최적 비율 찾기) **
- φ=1(FLOPS 2배) 고정 (모델 사이즈 결정 스케일링 계수, 기본 φ=0, 1배)
- grid search로 α* β^2 *γ^2=2 기준으로 여러 조합 시행.
  - 조합 1, 2, 3 의 accuracy 확인 결과
  → 조합3: α=1.2, β=1.1, γ=1.15일 때 (최적)
**STEP 2: (동일 비율 고정 후 크기 증가시키기.)**
α=1.2, β=1.1, γ=1.15일 때 (최적) 고정하고 φ 크기 조절
(α, β, γ에 φ를 제곱하는 구조를 사용하는 이유는 파라미터의 동일 비율 증가를 위함)
- d=α(1.2) ^ φ
- w= β(1.1) ^ φ
- r= γ(1.15) ^ φ
ex : B1: φ=1 → 2배, B2: φ=2 →4배 크기
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

고찰: 동일 배율로 증가 시킨다 가정해도 배율이 작을 때와 클 때 최적의 파라미터 구성이 다를 수도 있지않나?
→ 맞다. 하지만 거의 유사한 경향성을 보이고 단순한 제한 조건으로도 좋은 성능을 내는 방향성이다.
mobile net-v2 base로 변환 시킨 EfficientNet-B0 사용
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

동일 성능 대 model에서 훨씬 더 적은 params 를 확인 가능
