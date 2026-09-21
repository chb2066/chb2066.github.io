---
title: DiT
paper: Scalable Diffusion Models with Transformers
venue: ICCV 2023
link: https://arxiv.org/abs/2212.09748
claim: diffusion의 U-Net 백본을 Transformer로 교체하고 조건 주입을 adaLN-Zero로 처리하면 스케일에 따라 성능이 예측 가능하게 오른다.
tags: [Generative, Diffusion]
tier: main
date: 2025-05-28
draft: false
---

[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

## Abstract

**문제**
diffusion 모델의 백본은 대부분 U-Net계열을 사용. 다른 분야에서는 Transformer가 뛰어난 scailing 특성을 보임

**해결책**
latent 공간에서 동작하는 Transformer 백본(DiT) 을 만들고, 조건 주입 방식 네 가지를 비교해 adaLN-Zero 선택
모델 Gflops와 FID가 강하게 상관됨을 보여 스케일링 법칙이 그대로 적용됨을 확인

---

## 1. Introduction

**기존 생성 방식**
- **DDPM** - 노이즈를 점진적으로 더하고 그 역과정을 복원하며 이미지를 생성
- **VAE** - latent space로 압축했다가 복원. 평균과 표준편차를 다룸

**이 논문의 질문**
- diffusion에서 U-Net을 Transformer로 바꾸면 어떻게 되는가
- 그리고 그 스케일링 특성이 따라오는가

---

## 3. Diffusion Transformers

### 3.1 Preliminaries

**입력 요소와 전체 흐름**
1. 원본 이미지를 입력
1. 사전학습된 VAE Encoder 를 통과 → 기존에는 fc를 거쳤다면 여기서는 fc 없이 출력
1. latent z 를 뽑음
1. z 를 DiT 에 입력

즉, DiT는 픽셀이 아니라 latent 공간에서 동작. 이 부분은 Latent Diffusion과 동일

### 3.2 Diffusion Transformer Design Space

**Patchify**

- 입력은 latent image
- 패치화해서 일렬로 나열
- positional embedding으로 위치 정보를 더함
- 여기서 patch size p 가 설계 다이얼 → p 가 작을수록 토큰 시퀀스가 길어지고 Gflops가 늘어남
- 논문은 p ∈ {2, 4, 8} 을 비교 → 작은 패치가 일관되게 더 낮은 FID

**DiT Block** - adaLN-Zero(Adaptive Layer Norm - Zero, scale 파라미터를 0으로 초기화)를 쓴다.

*기존 방식*
- Layer norm은 γ, β 로 한 이미지 내의 채널을 정규화
- 시간 t 와 클래스 label c 는 단순히 더하는 방식으로 값의 분포를 바꿨음

*adaLN-Zero 방식*
- γ, β 가 t, c 에 의해 바뀌도록 하는 layer norm → 단순 값 변환이 아니라 feature map의 강도와 분포를 결정
- 메커니즘
  - z = Emb(t) + Emb(c)
  - MLP(z) = γ, β
  - adaLN(x, z) = γ(z) · LayerNorm(x) + β(z)

*블록 구조*
- adaLN-Zero를 통과하며 시간과 클래스 정보가 주입됨
- Self-Attention 으로 전역 정보를 얻고 noise 부분을 강조해 학습
- Pointwise MLP(각 패치에 대한 MLP)로 정보를 가공하고 업데이트

**Final Layer**
- Standard Layer Norm과 Linear로 데이터를 정리하고 차원을 맞춤
- unpatchify로 재배치
- conv로 최종 출력 → VAE latent z 의 예상 noise

**조건 주입 방식 비교** 

- In-context : t, c 임베딩을 추가 토큰 두 개로 시퀀스에 붙임. ViT의 cls token과 비슷. 마지막 블록 뒤에 제거
- Cross-attention : t, c 를 길이 2의 별도 시퀀스로 두고 self-attention 뒤에 cross-attention 층 추가
- adaLN : γ, β 를 직접 학습하지 않고 t, c 임베딩의 합에서 회귀
- adaLN-Zero : adaLN에 더해 블록을 항등함수로 초기화 

adaLN-Zero가 연산량 측면에서 가장 우수

**adaLN-Zero의 zero-init 장점**
- 각 블록의 마지막 batch norm scale을 0으로 초기화하면 대규모 학습이 빨라짐
- diffusion U-Net도 residual 연결 직전의 마지막 conv를 zero-init

**adaLN 계열 제약**
- 세 블록 설계 중 adaLN만이 모든 토큰에 같은 함수를 적용하도록 제한됨 → 조건이 공간적으로 균일하게 작용
- 클래스 label처럼 전역 조건에는 맞지만, 위치마다 다른 조건을 줘야 한다면 부적합

---

## 4. Experimental Setup

**모델 구성** - ViT 설정을 따라 층 수, hidden size, head 수를 함께 키운다.

| 모델 | Layers | Hidden size | Heads | Gflops (I=32, p=4) |
|---|---|---|---|---|
| DiT-S | 12 | 384 | 6 | 1.4 |
| DiT-B | 12 | 768 | 12 | 5.6 |
| DiT-L | 24 | 1024 | 16 | 19.7 |
| DiT-XL | 28 | 1152 | 16 | 29.1 |

patch size와 조합하면 0.3에서 118.6 Gflops까지 커버한다.

**학습과 생성**
- **학습** - 실제 노이즈와 DiT의 예측 noise 차이로 학습
- **생성** - DiT가 낸 noise를 latent z 에서 빼고 VAE decoder에 넣어 이미지 출력

**Classifier-Free Guidance**
- 조건(class label c)이 있는 예측과 없는 예측을 각각 구해 그 차이를 증폭하는 방향으로 노이즈 예측을 보정
- 조건에 더 충실하면서도 품질 높은 샘플이 나옴

---

## 5. Experiments

**핵심 관찰**
- 모델 Gflops가 늘면 FID가 꾸준히 내려감
- 모델을 키우는 것과 패치를 줄이는 것 양쪽 모두 효과가 있고, 둘 다 결국 "토큰당 연산량을 늘리는" 같은 방향

- 가장 큰 DiT-XL/2가 기존 U-Net 기반(ADM, LDM)을 전부 앞서면서 연산 효율도 좋음
- ImageNet 256×256 클래스 조건부 생성에서 FID 2.27 로 당시 최고 성능
- 샘플링 연산을 늘려도 모델 연산 부족을 메우지 못함 → 모델을 키우는 것이 근본적

