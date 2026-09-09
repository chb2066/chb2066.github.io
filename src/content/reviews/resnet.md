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

#### 1. introduction

깊은 신경망이 성능 향상에 중요하지만 너무 깊은 층은 학습이 잘 안되고 성능 감소가 일어날 수 있다.

![그림 1](/img/resnet/01.png)

따라서 residual learning개념 도입

#### 2. Related Work

**Residual Representations**
기존 이미지 인식 방법은 특징을 직접 학습하는 방식이 다수
Resnet의 잔차 기반 표현은 기존 방법보다 성능 향상에 도움이 됐다.

**Shortcut Connections**
네트워크 내부에서 일부 레이어를 건너뛰는 shortcut connection이 모델 학습을 돕는 데 사용됨.

![그림 2](/img/resnet/02.png)

#### 3. Deep Residual Learning

**Residual Learning, Identity Mapping by Shortcuts**
- H(x)=F(x)+x 에서 H(x)를 다음 블럭으로 보내고 H(x)로 손실 함수 계산.
- 네트워크는 잔차(Residual Function) F(x) = H(x) - x 를 학습.
- Shortcut connection을 사용하여 원본 입력에 잔차를 더하는 방식으로 네트워크를 구성.
장점:
- H(x)를 통한 역전파 시 기울기가 소실되지 않고 네트워크 전체로 전달 가능(∂L/∂x=1+∂f(x)/∂x)
- 원본 입력 x가 직접 다음 레이어로 전달되어 정보 손실이 최소화된다.
- 모델이 F(x)만을 수정하여 입력 x를 최적화 하는 것이 아닌 부족한 부분만 보정할 수 있다.

**Network Architectures**

![그림 3](/img/resnet/03.png)

#### Experiments

**ImageNet Classification**

![그림 4](/img/resnet/04.png)

Shortcut connection이 적용된 Resnet에서 기존 신경망보다 성능이 향상됨.

추가 실험 결과:
50-layer, 101-layer, 152-layer ResNet을 실험, 깊이가 깊어질수록 성능 향상 확인.
→ Residual Network에서 Depth 증가에 따른 성능 향상 효과

![그림 5](/img/resnet/05.png)

**Bottleneck 구조(ResNet-50/101/152)**
ResNet-50/101/152에서는 Bottleneck 구조(1x1 → 3x3 → 1x1)를 사용하여 연산량을 줄임
채널 축소→ 특징 학습→ 채널 확장

![그림 6](/img/resnet/06.png)

FLOPs=Cin×Cout×KH×KW×Hout×Wout×2(연산량=입력채널×출력채널×커널 크기×특징맵 크기×2)
Feature Map크기(가로, 세로)=n-f+2p/s +1으로 구함.

위 그림 기준 계산 결과:
BasicBlock
(64×64×3×3+64×64×3×3)×56×56=0.26억FLOPs
Bottleneck
(256×64×1×1+64×64×3×3+64×256×1×1)×56×56=0.15억FLOPs

Bottleneck 사용 시 FLOPs가 40% 감소, 성능 유지
→Resnet-50/101/152 같은 깊은 모델이 학습 가능
**CIFAR-10 and Analysis**

![그림 7](/img/resnet/07.png)

ResNet은 CIFAR-10 데이터셋에서도 매우 깊은 네트워크에서 성능을 유지
특이점: 1000 layers 이상에서도 작동 가능하지만 1202 같은 경우 overfitting이 일어나기에 최적의 깊이를 찾아야 한다.
Object Detection
Resnet이 이미지 분류 외에 객체 탐지와 같은 다른 비전 작업에서도 효과적이다.
