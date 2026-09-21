---
title: EfficientNet
paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
venue: ICML 2019
link: https://arxiv.org/abs/1905.11946
claim: depth, width, resolution을 따로 키우는 대신 고정된 비율로 함께 키우는 compound scaling이 같은 연산량에서 더 좋다.
take: 배율이 작을 때와 클 때 최적 비율이 다를 수 있지 않냐는 의문이 남지만, 경향성이 거의 유사해서 단순한 제약만으로도 좋은 방향을 준다.
tags: [Backbone, Model Scaling]
tier: main
date: 2025-04-02
draft: false
---

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

## Abstract

**문제**

CNN은 자원의 양과 성능이 비례하게 모델 크기를 올림.
-> 하지만 모델의 크기를 어떻게 키울지는 임의로 정해져 왔고, 보통 depth·width·resolution 중 하나만 늘림

**해결책**

- 세 축을 고정된 비율로 함께 키우는 compound scaling 제안
- 그 비율은 작은 baseline에서 한 번만 grid search로 찾고 큰 모델에 재사용

---

## 1. Introduction

**기존 상황**
- 자원을 더 쓸 수 있으면 성능이 좋아졌지만 너무 늘리면 오히려 나빠짐 → ResNet에서 depth를 과하게 늘렸을 때 성능 감소, skip connection 등으로 해결한 사례
- 그래서 기존에는 depth, width, resolution 중 하나만 늘림

**이 논문의 질문**
- 네트워크를 더 넓게, 더 깊게 하는 것과 자원의 밸런스가 어떤 조합일 때 최고 성능을 내는가

---

## 3. Compound Model Scaling

### 3.2 Scaling Dimensions

**세 개의 계수** - grid search로 depth, width, resolution을 파라미터로 두고 실험한다.

- **α** - depth 계수 (레이어 수)
- **β** - width 계수 (채널 수)
- **γ** - 입력 이미지의 해상도

건물에 비유하면 depth는 층 수, width는 각 층의 면적이다. width를 늘린다는 것은 같은 층 안에서 채널 수를 늘려 얻을 수 있는 특징 수를 늘린다는 뜻이다.

- 한 축만 키우면 금세 포화됨 → 세 축을 함께 봐야 하는 이유

### 3.3 Compound Scaling

**제약 조건: α · β² · γ² ≈ 2** - 지수가 왜 이렇게 붙는지가 이 논문의 기술적 핵심이다.

- 깊이를 2배로 하면 FLOPs가 2배 → 층이 두 배니까 그대로
- width나 resolution을 2배로 하면 FLOPs가 4배 → 채널 수는 입력과 출력 양쪽에 곱해지고, 해상도는 가로와 세로 양쪽에 곱해지기 때문

conv 연산이 계산 비용을 지배하므로 총 FLOPs는 대략 `(α · β² · γ²)^φ` 로 늘어난다. 여기서 α · β² · γ² ≈ 2 로 제약하면 어떤 φ 를 잡아도 총 FLOPs가 대략 2^φ 배 → φ 가 곧 "연산량을 몇 배로 쓸 것인가"의 다이얼이 된다.

**STEP 1. 최적 비율 찾기**
- φ = 1 로 고정 (FLOPs 2배에 해당. 기본값 φ = 0 이 1배)
- `α · β² · γ² ≈ 2` 제약 아래에서 grid search
- 결과 α = 1.2, β = 1.1, γ = 1.15 가 최적

**STEP 2. 비율을 고정하고 크기 키우기**
```text
d = α^φ = 1.2^φ
w = β^φ = 1.1^φ
r = γ^φ = 1.15^φ
```
- B1은 φ = 1 로 2배, B2는 φ = 2 로 4배
- φ 를 지수로 올리는 이유는 세 파라미터를 동일한 비율로 함께 증가시키기 위해

> 고찰 - 동일 배율로 증가시킨다고 해도, 배율이 작을 때와 클 때 최적의 파라미터 구성이 다를 수도 있지 않나?
>
> → 맞다. 하지만 거의 유사한 경향성을 보이고, 단순한 제한 조건만으로도 좋은 성능을 내는 방향성이다.

이 지적은 실제로 후속 연구에서 다뤄지는 지점이기도 하다. 논문은 비율을 작은 모델에서 한 번만 찾고 그대로 밀어붙이는데, 큰 모델에서 다시 탐색하면 비용이 감당이 안 되기 때문이다. 정확도보다 탐색 비용을 규모에서 분리한 것이 실질적 기여에 가깝다.

---

## 4. EfficientNet Architecture

model scaling은 층의 연산자 자체를 바꾸지 않으므로 좋은 baseline이 있어야 스케일링의 효과가 산다. 그래서 새 baseline을 직접 만든다.

**EfficientNet-B0**
- 주 구성 블록은 mobile inverted bottleneck(MBConv), 여기에 squeeze-and-excitation 추가
- FLOPs 목표 400M

| 단계 | 연산자 | 해상도 | 채널 | 반복 |
|---|---|---|---|---|
| 2 | MBConv1, k3×3 | 112×112 | 16 | 1 |
| 3 | MBConv6, k3×3 | 112×112 | 24 | 2 |
| 4 | MBConv6, k5×5 | 56×56 | 40 | 2 |
| 5 | MBConv6, k3×3 | 28×28 | 80 | 3 |
| 6 | MBConv6, k5×5 | 14×14 | 112 | 3 |
| 7 | MBConv6, k5×5 | 14×14 | 192 | 4 |
| 8 | MBConv6, k3×3 | 7×7 | 320 | 1 |
| 9 | Conv1×1 & Pooling & FC | 7×7 | 1280 | 1 |

![그림](/img/efficientnet/02.png)

**MBConv의 숫자 - 채널 확장 배수**
1. 채널을 n배(1 또는 6)로 확장
1. depthwise convolution 수행
1. 다시 압축

"채널 확장 → conv → 채널 압축" 구조. 병목을 거꾸로 뒤집은 형태라 inverted bottleneck이라 부른다.

MobileNetV2를 기반으로 변환한 것이 EfficientNet-B0이고, FLOPs 목표가 더 커서 원본보다 약간 크다.

---

## 5. Experiments

- EfficientNet-B7이 ImageNet top-1 84.3% 로 당시 최고 성능
- 같은 정확도의 GPipe와 비교하면 파라미터가 8.4배 적고(66M 대 556M) 추론이 6.1배 빠름
- B1은 기존 최고 모델보다 7.6배 작음
- 작은 쪽에서도 성립 - 약 6.8M 파라미터로 top-1 74.8%

동일 성능 대비 훨씬 적은 파라미터를 쓴다는 것이 그래프 전체에서 일관되게 나타난다.

- 5.1 MobileNet·ResNet에 compound scaling을 적용해도 한 축만 키우는 것보다 나음 → 규칙이 이 백본에만 국한되지 않음
- 5.3 전이 학습에서도 적은 파라미터로 높은 정확도

