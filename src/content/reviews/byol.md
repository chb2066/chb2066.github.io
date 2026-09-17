---
title: BYOL
paper: Bootstrap Your Own Latent
venue: NeurIPS 2020
link: https://arxiv.org/abs/2006.07733
claim: negative pair 없이 positive pair만으로도, EMA target과 predictor 비대칭 구조가 collapse를 막아준다.
tags: [Self-supervised, Non-contrastive]
tier: basic
date: 2025-06-18
draft: true
---

> 📄 [**Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning**](https://arxiv.org/abs/2006.07733) · NeurIPS 2020 · Grill, Strub, Altché et al.

## Abstract

**문제**
contrastive 계열은 **negative pair** 로 붕괴를 막음 → 큰 배치나 memory bank 가 필요하고, **negative 를 정의하기 어려운 도메인**에서는 쓰기 곤란
augmentation 선택에도 민감함

**해결책**
positive pair 만으로 학습하되, **online 쪽에만 predictor 를 두는 비대칭**과 **stop-gradient** 로 붕괴를 막음
target network 는 online 의 **EMA** - 자기 자신의 과거를 목표로 삼아 스스로를 끌어올림

---

## 1. Introduction

**기존 상황**
- MoCo·SimCLR 등 contrastive learning 은 **negative pair 와 positive pair 를 모두 사용** → 같은 것은 가깝게, 다른 것은 멀게
- 여기서 negative sample 이 하는 일은 **붕괴 방지** → 없으면 모든 표현을 같은 값으로 만드는 자명한 해가 생김
- 대신 negative 를 충분히 확보하려면 큰 배치나 queue 가 필요함

**negative 를 버리려는 이유**
- **negative sample 을 정의하기 어려운 도메인**이 많음
- 의료 영상에서 두 환자의 스캔이 정말 "다른 것"인지, 로봇 궤적에서 두 시점이 정말 무관한지는 자명하지 않음
- negative 에 의존하지 않는 방법은 그런 곳에서 결정적인 이점을 가짐

**본 논문의 기여**
1. **positive pair 만으로** 학습하는 BYOL 제안. negative sample 이 전혀 없음
1. ImageNet linear evaluation **74.3%**(ResNet-50)로 당시 contrastive 최고 방법들을 앞섬
1. 배치 크기와 augmentation 선택에 **덜 민감함**을 실험으로 확인

---

## 3. Method

### 3.1 Description of BYOL

![Figure 2](/img/byol/f2.png)

**입력 요소** - 동일한 이미지를 다르게 augmentation 한 두 view

**두 네트워크**

| | online network | target network |
|---|---|---|
| 구성 | encoder → projection head → **predictor** | encoder → projection head |
| predictor | **있음** | **없음** |
| 갱신 | loss 로 역전파 | **EMA 로 online 을 따라감** |
| gradient | 흐름 | **stop-gradient** |

**흐름**
1. 같은 이미지에서 두 개의 view 를 만듦
1. view 1 은 online network 를, view 2 는 target network 를 통과함
1. online network 는 projection 뒤에 **predictor**(MLP)를 한 번 더 거침
1. predictor 의 최종 출력 차원이 target 의 projection 출력 차원과 같으므로 **두 값을 직접 비교**할 수 있음
1. 그 차이를 loss 로 줄임

**비대칭이 핵심이다.** online 쪽에만 predictor 가 있고 target 쪽에는 없다. 양쪽이 대칭이면 두 출력이 같아지는 자명한 해로 무너진다.

**Loss** - MSE. 정확히는 정규화된 두 벡터의 평균제곱오차이고, 이는 cosine similarity 와 동등하다.

**target 갱신**
```text
ξ ← τ·ξ + (1 − τ)·θ

ξ: target 파라미터
θ: online 파라미터
τ: 모멘텀 계수 (1에 가까움)
```

### 3.2 Intuitions on BYOL's behavior

negative sample 없이 "두 출력을 가깝게 만들라"고만 하면 모든 것을 같은 값으로 내는 해가 존재한다. BYOL 이 그리로 가지 않는 이유는 두 가지가 함께 작용하기 때문이다.

**① target 의 느린 변화**
- target 은 EMA 로 갱신되므로 큰 영향을 주지 않고 조금씩만 변함
- online 이 target 을 쫓아가는 동안 target 은 거의 고정된 목표로 남음 → **다양한 정보를 계속 사용할 수 있음**

**② predictor 와 stop-gradient 의 비대칭**
- online 에만 predictor 가 있고 target 으로는 기울기가 흐르지 않음
- 이 비대칭 때문에 **두 네트워크가 같은 자명한 해로 동시에 수렴하지 못함**

**부트스트랩이라는 이름의 의미** - target network 는 online network 의 과거 버전이다. 즉 **자기 자신의 이전 상태를 목표로 삼아 스스로를 끌어올린다.** 외부 정답이 없는데도 학습이 진행되는 이유가 여기 있고, 논문 제목이 그것을 가리킨다.

### 3.3 Implementation details

**입력 요소**
- SimCLR 과 유사한 augmentation - random crop, flip, color jitter, grayscale, Gaussian blur, solarization
- 두 view 에 **서로 다른 augmentation 분포**를 적용
- projection·prediction head 는 모두 hidden layer 하나를 가진 MLP

---

## 4. Experimental evaluation

- **ImageNet linear evaluation 74.3%**(ResNet-50) - negative sample 을 쓰는 당시 최고 방법들을 앞섬
- 더 큰 ResNet 에서도 일관되게 향상
- semi-supervised, transfer, detection·segmentation 전이에서도 contrastive 기준선을 상회

---

## 5. Building intuitions with ablations

![Figure 3](/img/byol/f3.png)

- **배치 크기에 덜 민감함** - SimCLR 은 배치가 작아지면 성능이 크게 떨어지는데, BYOL 은 negative sample 에 의존하지 않으므로 그 영향이 작음
- **augmentation 선택에도 더 강건함** - contrastive 방법은 특정 augmentation(색상 왜곡 등)을 빼면 성능이 급락하는데 BYOL 은 덜 그러함
- **predictor 를 제거하면 붕괴** - 비대칭이 실제로 붕괴 방지의 핵심임을 확인
- **τ 를 0으로 두면**(target = online) 역시 붕괴. 느린 목표가 필요함

두 번째와 세 번째가 실용적으로 중요하다. 큰 배치를 감당할 수 없거나, 도메인에 맞는 augmentation 을 잘 모를 때 선택지가 된다.

---

## 정리

BYOL이 보여준 것은 **negative sample이 붕괴 방지의 유일한 방법은 아니라는 것**이다.

같은 것을 가깝게 만들되 무너지지 않게 하려면 어떤 형태의 비대칭이 필요한데, 그 비대칭을 **negative sample로 만들 수도 있고 구조로 만들 수도 있다.** BYOL은 후자를 택했다.

그리고 여기서 떼어 쓸 수 있는 부품이 두 개 나온다.

**첫째, EMA target.** "느리게 변하는 목표"라는 장치는 자기지도뿐 아니라 강화학습의 타깃 네트워크, knowledge distillation의 teacher 등 어디서나 재사용된다.

**둘째, predictor + stop-gradient 조합.** 이후 SimSiam이 여기서 EMA까지 빼도 된다는 것을 보인다. 즉 BYOL이 필수라고 여겼던 것 중 일부는 사실 없어도 됐다.

다만 대가가 있다. **붕괴 방지가 미묘한 균형에 의존해서 하이퍼파라미터에 민감하고, 왜 동작하는지에 대한 이론이 완전하지 않다.**

---

*그림은 모두 원 논문에서 가져왔다. Grill et al., [Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733), NeurIPS 2020.*
