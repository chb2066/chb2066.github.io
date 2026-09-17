---
title: V-JEPA 2
paper: "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"
venue: arXiv 2025
link: https://arxiv.org/abs/2506.09985
claim: 웹 규모 비디오로 사전학습한 인코더를 얼리고 action-conditioned predictor를 붙이면, 소량의 로봇 데이터로 계획이 가능해진다.
tags: [Self-supervised, Video, Robotics]
tier: basic
date: 2025-07-30
draft: true
---

> 📄 [**V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning**](https://arxiv.org/abs/2506.09985) · arXiv 2025 · Assran, Bardes, Fan et al.

## Abstract

**문제**
로봇이 계획을 세우려면 **행동이 상태를 어떻게 바꾸는지**를 아는 world model 이 필요함
그런데 그 모델을 학습시킬 **행동 라벨이 붙은 로봇 데이터는 비쌈**

**해결책**
행동 라벨이 **없는** 인터넷 비디오 100만 시간으로 물리 표현을 먼저 학습
그 인코더를 **얼리고**, 행동 조건 predictor 만 로봇 영상 **62시간 미만**으로 붙임
임베딩 공간에서 rollout 하며 MPC 로 행동 시퀀스를 최적화 → 처음 보는 로봇·환경에서 zero-shot 조작

---

## 1. Introduction

![Figure 1](/img/v-jepa-2/f1.png)

**분리가 설계의 출발점**
- 물리를 배우는 데는 **대량의 비디오**가 필요하지만 라벨 없이 얻을 수 있음
- 행동과 상태를 잇는 부분만 **로봇 데이터**가 필요한데, 그건 훨씬 적어도 됨
- 두 단계로 나누면 비싼 쪽을 최소화할 수 있음

**두 단계 구성**
1. **V-JEPA 2** - 행동 라벨 없는 인터넷 비디오로 시각 표현과 물리 직관을 학습
1. **V-JEPA 2-AC** - 얼린 인코더 위에 행동 조건 predictor 를 소량의 로봇 데이터로 붙임

---

## 2. V-JEPA 2: Scaling Self-Supervised Video Pretraining

![Figure 2](/img/v-jepa-2/f2.png)

### 2.1 Methodology

- V-JEPA 1과 같은 JEPA 방식 - 마스킹된 시공간 영역의 **잠재 표현**을 예측
- V-JEPA 1이 비디오의 본질적인 시각 표현을 학습했다면, **V-JEPA 2는 학습 스케일을 극대화해 라벨 없이 실제 물리 법칙을 반영하게 함**
- mask prediction 에서 **mask denoising**(L1 loss)으로 바뀐 것도 차이점

### 2.2 Scaling Self-Supervised Video Learning

![Figure 3](/img/v-jepa-2/f3.png)

네 가지 축을 함께 키운다.
- **데이터** - 사전학습 영상의 양과 다양성
- **모델** - encoder 파라미터 수
- **해상도** - 입력 프레임 크기
- **학습 스케줄** - iteration 수와 warmup 구조

각 개입의 기여를 따로 떼어 측정한 결과, **어느 하나만 키우면 금방 포화**하고 함께 키울 때만 이득이 이어진다.

### 2.3 Pretraining Dataset

**입력 요소**
- 인터넷 비디오 **100만 시간**
- 이미지 **100만 장**
- 여러 출처를 합쳐 **VideoMix22M** 구성

작은 데이터셋으로 학습한 같은 크기의 모델과 비교하면, 데이터 규모를 키운 쪽이 일관되게 낫다.

### 2.4 Pretraining Recipe

- 저해상도·짧은 클립으로 대부분을 학습하고, **후반에 고해상도와 긴 클립으로 적응**시키는 단계적 스케줄
- 학습 비용을 앞단에서 아끼고 dense·장기 의존이 필요한 능력만 뒤에서 확보

---

## 3. V-JEPA 2-AC: Learning an Action-Conditioned World Model

![Figure 6](/img/v-jepa-2/f6.png)

### 3.1 Action-Conditioned World Model Training

**입력 요소**
- 1단계에서 사전학습한 encoder - **얼린 상태로 사용**
- 새로 붙이는 predictor - 학습 대상
- Droid 데이터셋의 로봇 영상 **62시간 미만**, 로봇팔 움직임 라벨로 인과 관계를 학습

1단계의 100만 시간과 비교하면 극히 적은 양이다.

**손실 구성**
```text
Loss_total = Loss_teacher-forcing + Loss_rollout
```
둘 다 회귀 loss 다.

**Loss(teacher-forcing)** - 행동에 따른 **단기 상황 예측**
- `a_t` 는 액션, `s_t` 는 현재 상태, `E(x_t)` 는 현재 이미지를 인코더에 넣은 것. 출력은 `s_{t+1}` 이 무엇일지 예측하는 것
- 현재 상태 `s_t` 에 `a_t` 가 들어갔을 때의 출력 임베딩과, 이후 실제 데이터 `s_{t+1}` 의 임베딩 사이 차이를 최소화 → **임베딩 공간에서의 거리를 줄이는 것**이 목적

**Loss(rollout)** - 행동 시퀀스에 따른 **장기 상황 예측**
- 현재 상황 `s_t` 를 기준으로 predictor 에 `a_t` 를 넣어 나온 출력 `s_{t+1}` 을 다시 모델에 넣고 `a_{t+1}` 을 넣음. autoregressive 하게 T번 반복
- 결과적으로 현재 상황과 `a_t` 부터 `a_{t+T}` 까지의 액션만 보고 T번 이후의 상황을 예측하게 만듦
- 예측 임베딩과 실제 임베딩을 비교해 차이를 최소화

**두 loss를 함께 쓰는 이유** - teacher-forcing 만 쓰면 한 스텝 예측은 정확해도 여러 스텝을 이어 붙일 때 오차가 누적된다. rollout loss 가 **실제로 자기 예측을 다시 입력으로 받는 상황**을 학습에 포함시켜 그 누적을 억제한다.

### 3.2 Inferring Actions by Planning

![Figure 7](/img/v-jepa-2/f7.png)

목표는 **물건이 옮겨진 이미지를 보여주면 로봇이 그 상태로 만드는 것**이다.

1. 현재 보고 있는 이미지와 물건이 옮겨진 목표 이미지를 넣음
1. 두 이미지를 인코더에 넣어 임베딩 공간의 벡터로 변환함
1. predictor 가 회귀 기반이므로 여기서는 **행동 시퀀스** `(a_1, a_2, …, a_T)` 를 생성함
1. 이 시퀀스를 **CEM**(Cross-Entropy Method)으로 최적화함
1. `z_t` 에서 `a_{t+T}` 까지 rollout 해 예상 지점의 이미지 임베딩을 계산함
1. 계산한 임베딩과 실제 목표의 임베딩을 비교함
1. **실제 로봇의 움직임은 시퀀스 중 `a_1` 만 실행하고, 이후 다시 위 단계를 반복함**

마지막 단계가 중요하다. 계획을 세우고 첫 행동만 실행한 뒤 다시 계획하는 것은 **MPC의 receding horizon 과 정확히 같은 구조**다. 모델 오차가 여러 스텝에 걸쳐 누적되는 것을 매 스텝 재계획으로 보정한다.

---

## 4. Planning: Zero-shot Robot Control

![Figure 9](/img/v-jepa-2/f9.png)

**서로 다른 두 연구실의 Franka 팔에 zero-shot 으로 배포**해서, 이미지 목표만으로 물체를 집고 놓는 작업을 수행한다.

- **4.1 실험 설정** - 단일 목표 도달, 물체 잡기, 집어서 옮기기
- **4.2 결과** - 그 환경에서 학습하지 않았고 그 로봇으로 데이터를 모으지도 않았는데 동작함. 62시간 미만의 Droid 영상으로 학습한 predictor 가 다른 실험실의 다른 셋업으로 그대로 옮겨감
- **4.3 한계** - 목표를 이미지로만 지정해야 하고, 긴 지평의 과제에서는 재계획으로도 보정되지 않는 오차가 남음

에너지 지형을 그려 보면 목표 위치 근처에서 실제로 최솟값이 형성된다. 계획이 우연히 맞는 것이 아니라 predictor 가 만든 지형이 탐색을 이끌고 있다는 증거다.

---

## 정리

이 논문에서 가져갈 것은 **"무엇을 대량으로 배우고 무엇을 소량으로 붙일 것인가"의 분리**다.

물리 직관과 시각 표현은 인터넷 비디오로 라벨 없이 배울 수 있다. 반면 "이 행동을 하면 상태가 어떻게 변하는가"는 로봇 데이터가 필요하다. 그런데 **후자가 필요한 양은 전자보다 훨씬 적다.**

100만 시간과 62시간의 대비가 이 설계의 요점이다. 로봇 데이터가 비싸다는 병목을, 비싼 부분을 최소화하는 구조로 우회한다.

그리고 world model을 **생성이 아니라 계획에 쓴다**는 점이 중요하다. 비디오를 생성해서 계획하면 계산 비용이 크고, 평가도 예측의 충실도나 시각 품질로 흘러가기 쉽다. 임베딩 공간에서 rollout 하면 그 비용이 사라진다.

---

*그림은 모두 원 논문에서 가져왔다. Assran et al., [V-JEPA 2](https://arxiv.org/abs/2506.09985), arXiv 2025.*
