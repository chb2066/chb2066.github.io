---
title: AE, VAE
paper: Auto-Encoding Variational Bayes
venue: ICLR 2014
link: https://arxiv.org/abs/1312.6114
claim: latent를 점이 아니라 분포로 두고 KL로 정규화하면, 복원만 하던 AE가 새로운 샘플을 생성할 수 있게 된다.
tags: [Generative, Latent Variable]
tier: basic
date: 2025-06-02
draft: true
---

#### AE

데이터의 차원을 줄이면서 데이터의 주요 특징을 latent space로 압축하고 해당 특징을 다시 복원하면서 원본 이미지- 복원된 이미지=MSE를 통해서 이미지 복원을 하는 방식

#### VAE

- 기존 AE는 점 z를 통해 디코더로 복원하는 구조
- input 이미지
- encoder를 통해 압축 후 Loss를 통해 평균, 표준편차 구함.
- 평균, 표준편차를 통한 분포에서 z를 샘플링
VAE는 평균, 분산을 loss를 통해서 구하게 하고 그렇게 나온 분포에서 특정 z를 랜덤 샘플링한다.  개념 추가한 것. encoder로 평균과 분산 두 가지를 예측,
**VAE Loss**

```javascript
L = Reconstruction Loss + KL Divergence
- Reconstruction Loss = 복원된 이미지와 원본 이미지 간 차이 
- KL Divergence = D_KL(q(z|x) || p(z)) = D_KL(N(μ,σ²) || N(0,I))
	- q(z|x): encoder가 만든 분포 (평균 μ, 표준편차 σ)
	- p(z): 사전에 정해둔 목표 분포, 보통 표준정규분포 N(0,I)
	- 이 항이 encoder가 만드는 분포를 N(0,I)에 가깝게 정규화함
```

**KL term이 필요성 **
- Reconstruction loss만 쓰면 AE처럼 각 데이터가 latent space의 아무 곳에나 흩어져서 자기 자신만 잘 복원하면 되는 상태로 학습됨.
  → 학습 데이터에 없는 z를 샘플링하면 이상한 이미지가 나옴. (latent space에 "빈 공간"이 많아서)
- KL term이 모든 데이터의 분포를 원점 근처(N(0,I))로 모아주기 때문에, 학습 후에 z를 새로 랜덤 샘플링해도  그럴듯한 이미지가 나오는 생성 모델로서 기능.
- 즉, AE는 "압축&복원"만 하는 모델이고, VAE는 KL term 덕분에 새로운 샘플을 생성 가능한 모델이 됨.
**데이터 흐름 (Tensor Flow)**
조건:
- latent space를 100으로 설정
- 입력 이미지 x가 [32, 3, 256, 256] 라고 가정. (batch, channel, height, width)
과정:
1. 해당 이미지가 encoder를 통과
1. encoder 출력
  1. 출력 형태:
    1. mu: [32, 100], log_var: [32, 100]
1. 아래 수식과 같은 형태로 z를 출력 [32, 100]
  1. $z = \mu + \sigma \cdot \epsilon$
