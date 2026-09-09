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

기존 연구
- 기존 DDPM 은 노이즈를 만들고 그걸 복원하는 과정을 통해 이미지 생성.
- VAE는 Latent space를 사용하여 압축하고 그것을 복원하는 과정을 통해 이미지를 생성
  - 여기서는 평균, 표준편차 등을 사용함.
과정
1. 원본 이미지 입력
1. 사전 학습 된 VAE Encoder 통과(기존에는 fc를 거쳤다면 여기서는 fc없이 출)
1. Latent 추출(z)
1. DiT 입력 (z)

Method
- Patchify
  - **Input: **latent image I
  - 과정: 패치화해서 일렬로 나열
  - Positional embedding 해서 위치정보 더함
- DiT Block
  - adaLN-Zero(Adaptive Layer Norm - Zero, scale 파라미터를 0으로 초기화함):
    - 기존 방식
      - Layer norm은 γ, β 를 이용해 한 이미지 내의 채널을 사용하여 normalize 한다.
      - t, c(시간, 클래스 label) 등을 단순 더 하는 방식으로 사용하여 값의 분포를 바꿨다.
    - adaLN-Zero 방식
      - γ, β가 t, c에 의해 바뀌도록 하는 layer norm 방식을 사용해 단순 값 변환이 아닌 feature map의 강도, 분포 등을 결정하게 한다.
      - 매커니즘:
        - z= Emb(t) + Emb(c)
        - MLP(z) = γ, β
        - adaLN(x, z)= γ(z) * LayerNorm(x) + β (z)

![그림 1](/img/dit/01.png)

  - DiT block 구조:
    - adaLN-Zero 에 통과하여 해당 이미지의 시간과 클래스 정보가 입력됨.
    - Self-Attention 으로 전역 정보 획득 및 noise 부분 강조를 통한 학습.
    - Pointwise MLP(각 패치에 대한 MLP) 를 통해 정보 가공 후 업데이트
- Final Layer
  - Standard Layer Norm & Linear
    - 데이터 정리 및 차원 조절
  - unpatchify 로 재배치
  - conv로 최종 출력(VAE latent z 의 예상 nosie를 출력)
- 학습 vs 생성
  - 학습: 실제 노이즈 - DiT 의 예측 nosie 를 통해 학습
  - 생성: DiT에서 나온 noise를 latent z 에서 빼고 VAE decoder에 넣어서 사진 출력
  - Classifier-Free Guidance (CFG)
    - DiT 실험에서 생성 품질을 크게 끌어올리는 요소로 CFG를 사용함. 조건(class label c)이 있는 예측과 없는(unconditional) 예측을 각각 구해서, 그 차이를 증폭시키는 방향으로 노이즈 예측을 보정 → 조건에 더 충실하면서도 품질 높은 샘플 생성.
