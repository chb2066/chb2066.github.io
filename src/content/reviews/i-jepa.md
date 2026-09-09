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

핵심 키워드:
생성형이 아닌 SSL 접근법, 단일 context block사용, 같은 이미지에서 다양한 타깃 블럭에 대한 representation 예측
주요 전략:
masking 전략, 충분히 큰 규모의 샘플 타켓 블럭, 충분한 정보가 있는 context block을 사용
사용 가능 분야:
depth 추정, 오브젝트 카운팅, 선형 분류 등 다양한 곳에서 높은 성능

#### **배경 지식**

**기존 representation learning(Dinov2, SimCLR, ibot 등등)**
호환되는 입력 x, y에 대해서는 유사 임베딩을, 아닌 경우에 대해서는 다른 임베딩을 출력하도록 학습시키는 방식으로 표현 붕괴를 막는 기법 등을 사용함. contrastive loss, non-contrastive loss, 클러스터링 기반 접근법 등을 사용함.
**기존 Generative Architecture의 방법**
일반적인 생성형(z와 x의 정보를 둘 다 사용, ex: VAE)
1. Encoder에 x입력
1. decoder에 encoding x와 latent vector z 입력
  1. 1D 의 경우: x(512), z(8) 일 때 512+8=520 해서 두 개 concat한 [520] 벡터
    1. concat된 vector를 Transposed Convolution(up-sampling)
  1. 3D 의 경우: x=[16*16*512]일 때 z=[8]이면 H*W만큼 8에 해당하는 벡터를 복사하여 z=[16*16*8] 로 변환 후 concat한 [16*16*520] 벡터 생성
CV의 masking 기법
MAE, BEiT 같은 기법
1. 이미지 x와 일부를 마스킹한 이미지 내에 z(masking tokens과 위치정보)를 사용해 원본 이미지를 복원하게 하는 것

#### I-JEPA method

**Joint-Embedding Predictive Architectures**
- 기존
  - generative architecture는 입력 공간에 대해 손실 함수가 적용
- JPEA
  - 임베딩 공간에 손실함수가 적용.
    - 기존에는 이미지 등을 출력해서 이미지에 대해서 손실함수를 적용했다면 이건 출력 벡터를 실제 이미지를 target encoder에 넣어 나온 진짜 특징 벡터와 비교
    - 이를 통해 target encoder의 벡터와 context encoder에 넣어 나온 벡터를 비교
    - 같은 위치에 해당하는 벡터만 가져와서 비교하고 두 벡터 사이의 L2 distance 를 줄이도록 학습한다.
  - 추가 변수 z(위치)를 받는 predictor 네트워크를 사용
    - x로부터 z에 해당하는 y의 임베딩을 예측하도록 학습함.
  - loss는 context encoder의 가중치를 조정하게 함
  - 붕괴 방지를 위해 target encoder는 EMA로 조금씩 변경
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

#### I-JEPA 세부 아키텍처

> *[그림 자리 — Notion 원본에서 옮겨야 함]*

> 위 사진에 context는 tagets에 해당되는 부분을 임의로 제거한 입력될 이미지이고 자른 사각형은 context내 검은 외부 배경 부분을 제외한 것을 의미

**마스킹 전략 (Multi-Block Masking)**
입력 이미지 하나에 대한 문제(Context)와 정답(Target) 생성
- Target (맞춰야 할 곳): 이미지에서 4개 정도의 사각형 블록을 랜덤으로 선정(크기: 0.15~0.2, 종횡비: 0.75~1.5, 서로 겹칠 수 있음.)
- Context (힌트): 이미지에서 하나의 큰 사각형 블록을 선정(크기: 0.85~1.0)
  - 특이점: Target과 겹치는 부분은 Context에서 제거(모델이 정답 영역을 볼 수 없음)
    - 따라서 실제 픽셀 양은 0.85에서 target 이미지를 뺀 0.5~0.6정도가 남는다.
**Context Encoder (student)**
- 역할: 보여진 이미지 조각(Context)을 보고 특징 추출
- 구조: 표준 ViT (Vision Transformer)
- 입력: 마스킹 되고 남은 이미지 패치들
- 출력: 패치별 임베딩 벡터들 (Context Representation)
**Target Encoder (teacher)**
- 역할: 정답지(Target)의 임베딩을 생성
- 구조: Context Encoder와 동일한 구조의 ViT
- 업데이트: 학습되지 않고, Context Encoder의 가중치를 EMA(지수 이동 평균)로 업데이트
- 출력: Target 블록 위치에 해당하는 패치 임베딩 벡터들
**Predictor**
- 역할: Context 정보와 위치 정보를 받아서 Target 임베딩을 예측
- 구조: Encoder보다 가벼운 ViT를 사용
- 입력:
  - Context Encoder의 출력 (힌트 벡터)
  - Mask Token Target 블록의 위치 임베딩이 포함된 토큰.
- 출력: 예측된 임베딩 벡터.
**차별점**
SimCLR, BYOL 같은 view-invariance 기반 방법들이 다양한 aug를 사용하는 것과 대비됨.
MAE 같은 pixel_reconstruction 보다 학습 효율 높음.
counting, depth 등 다양한 downstream에서 사용 가능.

#### 세부 task를 위한 사용법

- 가정: ViT-Base 모델
- 조건: 벡터 차원 D=768, 패치 개수 N=196, 분류할 클래스 개수 K=1000
- Encoder 출력
  - 형태: [Batch_Size, 196, 768]
  - 각 패치마다 동일한 크기의 벡터, 각 패치에 대한 정보가 담긴 벡터 출력
- GAP
  - 각 벡터를 전부 더해서 패치 사이즈로 나눔(평균)
  - 형태: [Batch_Size, 768]
- Linear Layer
  - 동작: 행렬곱 (W*h+b)
    - W의 크기: [768, 1000]
  - 형태: [Batch_Size, 1000]
  - 결과: 각 클래스에 대한 logits
