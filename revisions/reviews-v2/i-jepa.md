---
title: I-JEPA
paper: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
venue: CVPR 2023
link: https://arxiv.org/abs/2301.08243
claim: 픽셀이 아니라 임베딩 공간에서 target 블록의 표현을 예측하면, 손으로 만든 augmentation 없이도 강한 표현이 학습된다.
tags: [Self-supervised, JEPA]
tier: main
date: 2025-06-24
draft: false
---

> 📄 [**Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**](https://arxiv.org/abs/2301.08243) · CVPR 2023 · Assran, Duval, Misra et al.

## Abstract

**문제**
불변성 기반 SSL 은 **손으로 만든 augmentation** 에 의존 → 그 변형이 무엇을 버려도 되는지 사람이 정해줘야 하고, 도메인이 바뀌면 다시 설계해야 함
생성 기반 SSL(MAE, BEiT)은 **픽셀을 복원**하므로 저수준 디테일에 용량이 쓰임

**해결책**
**임베딩 공간**에서 target 블록의 표현을 예측 - 손실을 입력 공간이 아니라 표현 공간에 적용
**multi-block masking** - 충분히 큰 target 블록 여러 개를, 정보가 충분한 단일 context 블록으로부터 예측
augmentation 없이도 선형 분류·depth·counting 등 폭넓은 downstream 에서 높은 성능

---

## 1. Introduction

![Figure 1](/img/i-jepa/f1.png)

**기존 representation learning (DINOv2, SimCLR, iBOT 등)**
- 호환되는 입력 `x`, `y` 에 대해서는 유사 임베딩을, 아닌 경우에는 다른 임베딩을 출력하도록 학습
- 표현 붕괴를 막기 위해 contrastive loss, non-contrastive loss, 클러스터링 기반 접근법 등을 사용
- 공통 전제 - **어떤 변형에 대해 불변이어야 하는가**를 사람이 정해줘야 함

**본 논문의 기여**
1. 손으로 만든 augmentation 없이, **임베딩 공간에서의 예측만으로** 강한 표현을 학습
1. 무엇이 필요한지를 분해 - **마스킹 전략**, **충분히 큰 규모의 target 블록**, **충분한 정보가 있는 context 블록**
1. MAE 계열 대비 **학습 효율이 높고**, depth·counting 같은 저수준 태스크에서도 동작

---

## 2. Background

![Figure 2](/img/i-jepa/f2.png)

**일반적인 생성형 구조** - `z` 와 `x` 의 정보를 둘 다 사용 (예: VAE)
1. Encoder 에 `x` 입력
1. decoder 에 encoding 된 `x` 와 latent vector `z` 입력
   - **1D** - `x`(512), `z`(8) 일 때 `512+8=520`, 두 개를 concat 한 `[520]` 벡터. concat 된 벡터를 Transposed Convolution(up-sampling)
   - **3D** - `x=[16*16*512]` 일 때 `z=[8]` 이면 `H*W` 만큼 8에 해당하는 벡터를 복사해 `z=[16*16*8]` 로 변환한 뒤 concat 해 `[16*16*520]` 벡터 생성

**CV 의 masking 기법** - MAE, BEiT 같은 기법
- 이미지 `x` 와 일부를 마스킹한 이미지 내에 `z`(masking token 과 위치정보)를 사용해 **원본 이미지를 복원**하게 하는 것
- 손실이 **입력 공간**에 적용된다는 것이 핵심적인 차이

---

## 3. Method

### Joint-Embedding Predictive Architecture

| | generative architecture | JEPA |
|---|---|---|
| 손실이 적용되는 곳 | **입력 공간**(픽셀) | **임베딩 공간** |
| 비교 대상 | 복원된 이미지 ↔ 원본 이미지 | 예측 벡터 ↔ target encoder 가 만든 벡터 |
| 버릴 정보 결정 | 못 버림 - 전부 복원해야 함 | **target encoder 가 알아서 정함** |

- 기존에는 이미지 등을 출력해서 이미지에 대해 손실함수를 적용했다면, 여기서는 출력 벡터를 **실제 이미지를 target encoder 에 넣어 나온 진짜 특징 벡터**와 비교
- **같은 위치에 해당하는 벡터만 가져와서** 비교하고 두 벡터 사이의 **L2 distance** 를 줄이도록 학습
- 추가 변수 `z`(위치)를 받는 **predictor** 네트워크를 사용 → `x` 로부터 `z` 에 해당하는 `y` 의 임베딩을 예측
- loss 는 **context encoder 의 가중치를 조정**하게 함
- 붕괴 방지를 위해 target encoder 는 **EMA** 로 조금씩 변경

![그림](/img/i-jepa/01.png)

### 세부 아키텍처

![Figure 3](/img/i-jepa/f3.png)

![그림](/img/i-jepa/02.png)

> 위 그림에서 context 는 target 에 해당되는 부분을 임의로 제거한 입력 이미지이고, 자른 사각형은 context 내 검은 외부 배경 부분을 제외한 것을 의미한다.

**마스킹 전략 (Multi-Block Masking)** - 입력 이미지 하나에 대한 문제(Context)와 정답(Target) 생성

![Figure 4](/img/i-jepa/f4.png)

- **Target**(맞춰야 할 곳) - 이미지에서 4개 정도의 사각형 블록을 랜덤 선정. 크기 0.15~0.2, 종횡비 0.75~1.5, 서로 겹칠 수 있음
- **Context**(힌트) - 이미지에서 하나의 큰 사각형 블록을 선정. 크기 0.85~1.0
  - **특이점** - Target 과 겹치는 부분은 Context 에서 제거. 모델이 정답 영역을 볼 수 없음
  - 따라서 실제 픽셀 양은 0.85 에서 target 이미지를 뺀 **0.5~0.6 정도**가 남음

**Context Encoder (student)**
- **역할** - 보여진 이미지 조각(Context)을 보고 특징 추출
- **구조** - 표준 ViT
- **입력** - 마스킹되고 남은 이미지 패치들
- **출력** - 패치별 임베딩 벡터들 (Context Representation)

**Target Encoder (teacher)**
- **역할** - 정답지(Target)의 임베딩을 생성
- **구조** - Context Encoder 와 동일한 구조의 ViT
- **업데이트** - 학습되지 않고, Context Encoder 의 가중치를 **EMA** 로 따라감
- **출력** - Target 블록 위치에 해당하는 패치 임베딩 벡터들

**Predictor**
- **역할** - Context 정보와 위치 정보를 받아서 Target 임베딩을 예측
- **구조** - Encoder 보다 가벼운 ViT
- **입력** - Context Encoder 의 출력(힌트 벡터), 그리고 Target 블록의 위치 임베딩이 포함된 **Mask Token**
- **출력** - 예측된 임베딩 벡터

**차별점**
- SimCLR, BYOL 같은 **view-invariance 기반 방법들이 다양한 augmentation 을 사용하는 것과 대비**됨
- MAE 같은 **pixel reconstruction 보다 학습 효율이 높음**
- counting, depth 등 다양한 downstream 에서 사용 가능

---

## 5. Image Classification

![Table 1](/img/i-jepa/t1.png)

- ImageNet-1K 선형 평가에서 augmentation 기반 방법들과 경쟁하거나 앞섬
- 라벨 1% 만 쓰는 준지도 설정에서도 강함

**세부 task 를 위한 사용법**

**입력 요소** - ViT-Base 기준. 벡터 차원 `D=768`, 패치 개수 `N=196`, 분류할 클래스 개수 `K=1000`

1. **Encoder 출력** - 형태 `[Batch_Size, 196, 768]`. 각 패치마다 동일한 크기의 벡터, 각 패치에 대한 정보가 담긴 벡터를 출력
1. **GAP** - 각 벡터를 전부 더해서 패치 수로 나눔(평균). 형태 `[Batch_Size, 768]`
1. **Linear Layer** - 행렬곱 `W·h + b`, `W` 의 크기는 `[768, 1000]`. 형태 `[Batch_Size, 1000]`
1. **결과** - 각 클래스에 대한 logits

---

## 6. Local Prediction Tasks

- object counting, depth prediction 같은 **저수준·국소 태스크**에서 평가
- 불변성 기반 방법들은 augmentation 으로 이런 정보를 일부러 버리기 때문에 약함
- I-JEPA 는 그런 변형을 쓰지 않으므로 **위치·수량 정보가 표현에 남아 있음**

---

## 7. Scalability

- 모델 크기와 데이터 규모를 함께 키우며 측정 → 둘 다 늘릴 때 성능이 계속 오름
- MAE 대비 **더 적은 연산으로 같은 성능**에 도달

---

## 8. Predictor Visualizations

![Figure 6](/img/i-jepa/f6.png)

- predictor 의 출력을 디코딩해서 시각화 → 예측이 **위치와 대략적인 형태를 담고 있음**
- 픽셀 수준의 세부는 담지 않음. **무엇을 버렸는지가 그림에 그대로 드러난다**

---

## 9. Ablations

![Table 6](/img/i-jepa/t6.png)

- **target 블록이 너무 작으면** 과제가 너무 쉬워져 표현이 약해짐
- **context 블록의 정보가 부족하면** 예측이 성립하지 않음
- 즉 이 방법이 통하려면 **"충분히 어렵되 풀 수 있는" 예측 과제**로 맞춰야 함

---

## 정리

I-JEPA가 보여준 것은 **손실을 어느 공간에 적용하는가가 표현의 성격을 결정한다**는 것이다.

픽셀 공간에서 복원을 시키면 모든 세부를 맞혀야 하므로 용량이 저수준 디테일로 흘러간다. 임베딩 공간에서 예측을 시키면 **무엇을 버릴지를 target encoder가 스스로 정한다.** 사람이 augmentation으로 "이건 버려도 된다"를 지정해주던 일을 모델이 대신하는 셈이다.

여기서 옮겨갈 만한 재료는 두 가지다.

**첫째, 불변성을 augmentation이 아니라 예측 과제로 유도하기.** 도메인마다 augmentation을 다시 설계하지 않아도 되고, 그래서 비디오(V-JEPA)나 로봇(V-JEPA 2)으로 그대로 옮겨간다.

**둘째, 예측 과제의 난이도를 설계 변수로 다루기.** target을 너무 작게 잡으면 쉬워지고, context를 너무 줄이면 불가능해진다. ablation이 보여주는 것은 이 두 경계 사이를 맞추는 일이 방법의 본체라는 점이다.

---

*`f`·`t` 로 시작하는 그림은 원 논문에서 가져왔다. Assran et al., [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243), CVPR 2023.*
