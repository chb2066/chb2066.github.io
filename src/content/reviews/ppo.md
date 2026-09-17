---
title: PPO
paper: Proximal Policy Optimization Algorithms
venue: arXiv 2017
authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
link: https://arxiv.org/abs/1707.06347
claim: 정책 비율을 잘라 갱신 폭을 제한하면, 2차 최적화 없이도 신뢰영역의 효과를 값싸게 얻는다.
tags: [RL, Policy Optimization]
tier: basic
date: 2025-04-16
draft: true
---

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

## Abstract

**문제**
정책 경사 방법은 한 번의 갱신이 너무 커지면 정책이 망가짐. 망가진 정책으로 수집한 데이터가 다시 학습을 망쳐 회복이 어려움
TRPO 는 KL 제약으로 이를 풀었지만 2차 최적화가 필요해 구현이 복잡하고 비쌈

**해결책**
정책 비율을 잘라내는(clip) 대리 목적함수 - 범위를 벗어나면 더 가봐야 목적함수가 개선되지 않게 만듦
1차 최적화만으로 신뢰영역의 효과를 얻고, 같은 배치를 여러 epoch 재사용할 수 있어 표본 효율도 개선

---

## 1. Introduction

**policy 갱신이 실제로 바꾸는 것** - Transformer 기반 정책이라면 갱신되는 것은 다음과 같다.
- Self-Attention 가중치(Q, K, V 행렬) - query, key, value 의 가중치가 갱신됨
- **Feed-Forward 가중치** - FFN 에서 GELU 같은 비선형 함수로 사라지는 값을 조절할 수 있음. 가중치를 더하거나 빼서 특정 성질에 대한 벡터값이 바뀌면, 역전파를 통해 중요도에 따른 제외나 강조가 가능해짐
- **Layer Normalization 파라미터** - 각 feature 에 대한 정규화를 통해 위와 같이 특징별 중요도 조절이 가능해짐
- **Positional Embedding** - 토큰 단위로 위치가 임베딩돼 있다고 하면, 각 토큰의 임베딩이 더 관련 있는 토큰과 가까워지도록 조정됨. 의미론적 표현이 증대됨

즉 "정책을 갱신한다"는 것은 이 파라미터들을 보상을 최대화하는 방향으로 조금씩 움직이는 것이다.

```python
# 보상을 최대화하는 방향으로
gradient = ∇_θ (보상 - KL페널티)

# 각 파라미터를 조금씩 수정
새_가중치 = 기존_가중치 + learning_rate * gradient

# 너무 큰 변화 방지
if 변화량 > clip_범위:
    변화량 = clip_범위
```

**본 논문의 기여**
1. clipping 만으로 신뢰영역 효과를 내는 clipped surrogate objective 제안
1. KL 계수를 자동 조정하는 adaptive KL penalty 변형도 함께 제시·비교
1. 연속 제어와 Atari 양쪽에서 기존 방법 대비 우수한 sample complexity 를 확인

---

## 2. Background: Policy Optimization

### 2.1 Policy Gradient Methods

- 기대 보상의 기울기를 추정해 정책 파라미터를 직접 갱신
- 온폴리시라 한 번 쓴 데이터를 버려야 함 → 표본 효율이 낮음
- 학습률을 크게 잡으면 정책이 급격히 바뀌어 붕괴하기 쉬움

### 2.2 Trust Region Methods

- TRPO 는 KL 발산에 대한 제약으로 갱신 폭을 직접 제한
- 제약을 정확히 풀려면 2차 최적화(conjugate gradient + line search)가 필요 → 구현이 복잡하고 비쌈
- PPO 는 같은 효과를 1차 최적화로, 목적함수를 자르는 것만으로 얻는다

---

## 3. Clipped Surrogate Objective

정책 비율을 `r(θ) = π_θ(a|s) / π_old(a|s)` 로 두면, 목적함수는 이렇게 된다.

```text
L = E[ min( r(θ)·A,  clip(r(θ), 1-ε, 1+ε)·A ) ]
```

`A` 는 advantage 다. 비율이 `[1-ε, 1+ε]` 범위를 벗어나면 잘라내므로, 그 방향으로 더 가봐야 목적함수가 개선되지 않는다. 자연스럽게 갱신 폭이 제한된다.

- advantage 가 양수일 때와 음수일 때 clipping 이 작용하는 쪽이 반대가 됨
- 여러 목적함수를 한 번의 갱신 경로를 따라 그려보면, clipped 목적함수만 지나친 갱신에 대해 보상을 끊음

---

## 4. Adaptive KL Penalty Coefficient

- 대안으로, KL 페널티를 목적함수에 더하고 계수를 매 갱신마다 자동 조정하는 변형도 제시
- 측정된 KL 이 목표보다 크면 계수를 키우고, 작으면 줄임
- 실험에서는 clipped 목적함수가 더 나았지만 비교 기준선으로 함께 제시

---

## 5. Algorithm

**입력 요소**
- N개의 병렬 액터
- 액터당 T 스텝 분량의 궤적
- advantage 추정(GAE)

**전체 절차**
1. N개의 병렬 액터로 데이터를 수집함
1. 수집된 데이터로 일정 epoch 동안 minibatch 학습으로 정책을 최적화함
1. 정책을 갱신함
1. clipping 으로 정책 변화가 너무 커지지 않게 막음

![그림](/img/rlhf/01.png)

**동작 방식**

![그림](/img/rlhf/02.png)

1. 병렬 액터가 환경을 동시에 실행함. 각 환경은 독립적으로 에이전트와 상호작용함
1. 현재 상태 `S_t` 와 보상 `R_t` 를 에이전트에게 전달함
1. 에이전트가 PPO 알고리즘을 실행함 - 데이터를 수집하고, 일정 epoch 동안 minibatch 로 최적화하고, 정책을 갱신함
1. 갱신된 정책으로 행동 `A_t` 를 결정함. 모든 환경에서 같은 정책을 쓰지만 행동은 다를 수 있음
1. 반복함

데이터를 여러 epoch 재사용할 수 있는 이유 - 일반적인 정책 경사는 온폴리시라 한 번 쓴 데이터를 버려야 한다. PPO 는 비율 `r(θ)` 로 중요도를 보정하고 clipping 으로 안전 범위를 강제하므로, 같은 배치를 여러 epoch 돌려도 무너지지 않는다. 샘플 효율이 개선되는 지점이다.

---

## 6. Experiments

- 6.1 여러 대리 목적함수 비교 - clipping 이 KL 페널티 변형들보다 나음
- 6.2 MuJoCo 연속 제어에서 A2C·TRPO·CEM 등을 상회
- 6.3 Roboschool 휴머노이드 주행·방향 전환 같은 고난도 과제도 학습

- 6.4 Atari 49개 게임에서 A2C 대비 대부분 우세

---

## 정리

PPO에서 가져갈 것은 알고리즘 자체보다 "갱신 비율을 잘라 안전 범위를 강제한다"는 형태다.

이 골격은 반복적으로 갱신되는 어떤 시스템에도 붙일 수 있다. 신뢰영역을 2차 최적화로 정확히 계산하는 대신, 목적함수를 잘라서 더 가봐야 이득이 없게 만드는 것이다. 근사이지만 구현이 단순하고 실제로 잘 동작한다.

다만 대가가 있다. 하이퍼파라미터에 민감하고 구현 세부가 성능을 좌우한다. 정규화 방식, clipping 범위, 가치 함수 clipping 여부 같은 것들이다. 재현성 문제가 잘 알려져 있는 알고리즘이기도 하다.

