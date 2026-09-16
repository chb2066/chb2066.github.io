---
title: ResNet
paper: Deep Residual Learning for Image Recognition
venue: CVPR 2016
link: https://arxiv.org/abs/1512.03385
claim: shortcut connection으로 잔차 F(x)=H(x)-x만 학습하게 하면 깊은 망에서도 기울기가 살아남는다.
tags: [Backbone, CNN]
tier: basic
date: 2025-03-01
draft: true
---

> 📄 [**Deep Residual Learning for Image Recognition**](https://arxiv.org/abs/1512.03385) · CVPR 2016 · He, Zhang, Ren, Sun

## Abstract

**문제**
망을 깊게 할수록 성능이 좋아진다고 알려져 있었는데, 어느 지점부터는 **더 깊게 하면 오히려 나빠짐**. 그것도 과적합이 아니라 학습 오차부터 올라감

**해결책**
층이 전체 사상 `H(x)`를 학습하는 대신 **입력과의 차이 `F(x) = H(x) − x`만 학습**하게 바꿈
shortcut connection으로 입력을 그대로 더해줘서 항등 사상을 기본값으로 만듦

---

## 1. Introduction

**degradation 문제**

![Figure 1](/img/resnet/f1.png)

- 너무 깊은 층은 학습이 잘 되지 않고 성능이 오히려 감소함
- **이것은 과적합이 아님** → 과적합이라면 학습 오차는 계속 내려가고 테스트 오차만 올라가야 하는데, 실제로는 **학습 오차부터 올라감**
- 즉 최적화 자체가 안 되는 문제

**논리적으로는 이상한 상황**
- 깊은 망은 얕은 망에 항등 사상 층을 얹은 것과 같아야 함 → 최소한 얕은 망만큼은 나와야 함
- 그런데 그렇지 않음 → **최적화 알고리즘이 항등 사상을 찾지 못한다는 뜻**

**본 논문의 기여**
1. 잔차를 학습하게 바꿔 항등 사상을 기본값으로 만듦 → 깊이를 늘려도 나빠지지 않음
1. 152층까지 학습시켜 ImageNet top-5 오류율 3.57%, ILSVRC 2015 분류 1위
1. VGG-19보다 8배 깊으면서 복잡도는 더 낮음

---

## 2. Related Work

**Residual Representations**
- 기존 이미지 인식은 특징을 직접 학습하는 방식이 다수
- 잔차 기반 표현이 기존 방법보다 성능 향상에 도움이 됨

**Shortcut Connections**
- 네트워크 내부에서 일부 레이어를 건너뛰는 연결은 이전에도 모델 학습을 돕는 데 쓰였음

![그림](/img/resnet/02.png)

---

## 3. Deep Residual Learning

### 3.1 Residual Learning

![Figure 2](/img/resnet/f2.png)

**입력 요소와 흐름**
- `H(x) = F(x) + x` 에서 `H(x)`를 다음 블록으로 보내고, `H(x)`로 손실 함수를 계산
- 네트워크는 **잔차 함수 `F(x) = H(x) − x`를 학습**
- shortcut connection으로 원본 입력에 잔차를 더해 구성

**이 형태가 주는 것 세 가지**

1. **기울기가 소실되지 않음** — `H(x)`를 통한 역전파에서
   ```text
   ∂L/∂x = 1 + ∂F(x)/∂x
   ```
   더해진 `1` 때문에 기울기가 0이 되지 않음
1. **정보 손실 최소화** — 원본 입력 `x`가 직접 다음 레이어로 전달됨
1. **부족한 부분만 보정** — `x`를 통째로 최적화하는 게 아니라 `F(x)`만 수정하면 됨

세 번째가 degradation에 대한 직접적인 답이다. **항등 사상이 필요하면 `F(x)`를 0으로 만들면 되고, 그건 쉽다.**

### 3.3 Network Architectures

![Table 1](/img/resnet/t1.png)

![그림](/img/resnet/03.png)

**Bottleneck 구조** — ResNet-50/101/152에서는 `1×1 → 3×3 → 1×1`로 연산량을 줄인다. 채널 축소 → 특징 학습 → 채널 확장의 순서.

![Figure 5](/img/resnet/f5.png)

```text
FLOPs = Cin × Cout × KH × KW × Hout × Wout × 2
      = 입력채널 × 출력채널 × 커널 크기 × 특징맵 크기 × 2

Feature Map 크기 = (n - f + 2p) / s + 1
```

| 구조 | 계산 | FLOPs |
|---|---|---|
| BasicBlock | (64×64×3×3 + 64×64×3×3) × 56×56 | 0.26억 |
| Bottleneck | (256×64×1×1 + 64×64×3×3 + 64×256×1×1) × 56×56 | **0.15억** |

- **FLOPs가 40% 감소하면서 성능은 유지됨** → 그래서 ResNet-50/101/152 같은 깊은 모델의 학습이 가능해짐

---

## 4. Experiments

### 4.1 ImageNet Classification

![그림](/img/resnet/04.png)

- shortcut connection이 적용된 ResNet이 기존 신경망보다 성능이 향상됨
- 50층, 101층, 152층을 실험한 결과 **깊이가 깊어질수록 성능이 향상** → residual network에서는 깊이 증가가 실제로 성능 향상으로 이어짐

![Table 3](/img/resnet/t3.png)

- **ImageNet test set top-5 오류율 3.57%** — ILSVRC 2015 분류 부문 1위
- **152층 ResNet이 VGG-19보다 8배 깊으면서도 복잡도는 더 낮음**

### 4.2 CIFAR-10 and Analysis

![그림](/img/resnet/07.png)

- CIFAR-10에서도 매우 깊은 네트워크에서 성능이 유지됨
- **특이점** — 1000층 이상에서도 작동은 하지만 1202층 같은 경우 **과적합이 일어남** → 최적의 깊이를 찾아야 한다는 뜻

여기서 degradation과 과적합이 분리된다. 잔차 연결이 degradation은 해결했지만, **깊이를 무한정 늘리면 이번엔 진짜 과적합이 나타난다.** 두 문제는 다른 것이다.

### 4.3 Object Detection

- COCO 객체 검출에서 상대적으로 **28% 개선**
- ILSVRC와 COCO 2015의 검출, localization, segmentation 부문에서도 1위

---

## 정리

ResNet이 남긴 것은 특정 아키텍처가 아니라 **`y = x + F(x)` 라는 형태 자체**다.

이 형태의 값어치는 **항등함수가 기본값이 된다**는 데 있다. 블록을 추가해도 최소한 나빠지지는 않으므로, 모듈을 안전하게 덧붙일 수 있다. 그리고 `F`를 0으로 초기화하면 **사전학습 모델을 건드리지 않고 확장**할 수 있다.

LoRA, adapter, ControlNet, adaLN-Zero가 전부 같은 골격이다. "**zero-init 잔차 가지로 새 능력을 붙인다**"는 패턴은 이 논문에서 시작해서 지금도 계속 재사용되고 있다.

부수적으로 얻은 것도 있다. 잔차 스트림이 **각 층이 읽고 쓰는 공유 작업공간**이 되면서, 이후 해석가능성 연구에서 개입의 무대가 됐다.

---

*`f`·`t` 로 시작하는 그림은 원 논문에서 가져왔다. He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), CVPR 2016.*
