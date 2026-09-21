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

[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## Abstract

**문제**
연속 잠재변수를 가진 방향성 확률 모델에서 사후분포 `p(z|x)` 가 계산 불가능함
기존 변분 추론은 데이터마다 최적화를 돌려야 해서 대규모 데이터셋에 쓰기 어려움

**해결책**
사후분포를 신경망 encoder 로 근사하고, 변분 하한(ELBO)을 직접 최적화
reparameterization trick 으로 샘플링 연산에 기울기를 흘려 역전파 가능하게 만듦
결과물이 VAE - 복원만 하던 AE 가 새로운 샘플을 생성할 수 있게 됨

---

## 1. Introduction

**AE 란**
- 데이터의 차원을 줄이면서 주요 특징을 latent space 로 압축하고, 그 특징을 다시 복원하는 방식
- 원본 이미지와 복원된 이미지의 MSE 로 학습함
- 압축과 복원만 함. 그 이상은 하지 않음

**기존 상황**
- 연속 잠재변수 모델의 사후분포가 계산 불가능 → MCMC 는 느리고, 평균장 변분 추론은 데이터포인트마다 최적화가 필요
- 큰 데이터셋에서는 둘 다 감당이 안 됨

**본 논문의 기여**
1. 변분 하한의 재파라미터화된 추정량(SGVB)을 제시 - 표준 SGD 로 최적화 가능
1. i.i.d. 데이터셋에 대해 인식 모델(encoder)을 함께 학습하는 AEVB 알고리즘 제안
1. 그 특수한 경우로 VAE 를 구성

---

## 2. Method

### 2.1 Problem scenario

**입력 요소**
- 관측 데이터 `x` 와 관측되지 않는 잠재변수 `z`
- 생성 모델 `p(z)·p(x|z)` - 디코더에 해당
- 인식 모델 `q(z|x)` - 계산 불가능한 사후분포 `p(z|x)` 를 근사하는 encoder

**AE 와의 차이** - 기존 AE 는 점 `z` 를 디코더에 넣어 복원한다. VAE 는 여기에 확률을 넣는다.

1. 입력 이미지가 들어감
1. encoder 를 통과해 평균 `μ` 와 표준편차 `σ` 를 출력함
1. 그 평균과 표준편차가 정의하는 분포에서 `z` 를 샘플링함
1. 디코더가 `z` 로부터 복원함

핵심은 encoder 가 평균과 분산 두 가지를 예측한다는 것이고, 그렇게 나온 분포에서 특정 `z` 를 랜덤 샘플링한다는 점이다.

### 2.2 The variational bound

```text
L = Reconstruction Loss + KL Divergence
```

- **Reconstruction Loss** - 복원된 이미지와 원본 이미지 간의 차이
- **KL Divergence** - `D_KL(q(z|x) || p(z)) = D_KL(N(μ, σ²) || N(0, I))`
  - `q(z|x)` - encoder 가 만든 분포. 평균 `μ`, 표준편차 `σ`
  - `p(z)` - 사전에 정해둔 목표 분포. 보통 표준정규분포 `N(0, I)`
  - 이 항이 encoder 가 만드는 분포를 `N(0, I)` 에 가깝게 정규화함

**식의 유도** - 임의로 두 항을 더한 게 아니다. 변분 추론에서 로그 가능도 `log p(x)` 의 하한(ELBO)을 유도하면 정확히 이 형태가 나온다. 직접 최대화할 수 없는 로그 가능도 대신, 계산 가능한 하한을 최대화하는 것이다.

**KL term 이 필요한 이유** - 이 부분이 VAE 를 이해하는 핵심이다.
- Reconstruction loss 만 쓰면 AE 처럼 각 데이터가 latent space 의 아무 곳에나 흩어져서, 자기 자신만 잘 복원하면 되는 상태로 학습됨
- 그러면 학습 데이터에 없는 `z` 를 샘플링했을 때 이상한 이미지가 나옴 → latent space 에 "빈 공간" 이 많기 때문
- KL term 이 모든 데이터의 분포를 원점 근처(`N(0, I)`)로 모아줌 → 학습 후에 `z` 를 새로 랜덤 샘플링해도 그럴듯한 이미지가 나오는, 생성 모델로서 기능하게 됨

| | 하는 일 |
|---|---|
| AE | 압축과 복원만 한다 |
| VAE | KL term 덕분에 새로운 샘플을 생성할 수 있는 모델이 된다 |

### 2.3 The SGVB estimator and AEVB algorithm

- 하한을 미니배치 단위의 몬테카를로 추정량으로 바꿔 SGD 로 최적화
- 데이터포인트마다 별도 최적화를 돌리지 않고 encoder 파라미터 하나를 전 데이터에 공유(amortized inference)
- 그 결과 대규모 데이터셋에서도 학습이 가능해짐

### 2.4 The reparameterization trick

여기서 문제가 하나 생긴다. 샘플링 연산에는 기울기가 흐르지 않는다. `z ~ N(μ, σ²)` 를 그대로 두면 역전파로 encoder 를 학습시킬 수 없다.

해결은 확률성을 분리하는 것이다.

$$
z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim N(0, I)
$$

랜덤성을 `ε` 이라는 외부 입력으로 빼내면, `μ` 와 `σ` 에 대해서는 결정론적인 연산만 남는다. 그래서 기울기가 흐른다.

이 트릭은 VAE 밖에서도 반복해서 등장한다. 이산 선택을 미분 가능하게 만드는 Gumbel-softmax 도 같은 발상이다.

---

## 3. Example: Variational Auto-Encoder

조건을 정해놓고 데이터 흐름을 따라가 본다.

**입력 요소**
- latent space 차원을 100 으로 설정
- 입력 이미지 `x` 가 `[32, 3, 256, 256]` (batch, channel, height, width)

**과정**
1. 이미지가 encoder 를 통과함
1. encoder 가 `mu`: `[32, 100]`, `log_var`: `[32, 100]` 을 출력함
1. 아래 수식으로 `z` 를 만듦. 형태는 `[32, 100]`

$$
z = \mu + \sigma \cdot \epsilon
$$

**`log_var` 를 출력하는 이유** - 분산은 항상 양수여야 하는데, 네트워크 출력에 그 제약을 걸기가 번거롭다. 로그를 예측하고 지수를 취하면 자동으로 양수가 되고 수치적으로도 안정적이다.

---

## 5. Experiments

- 2차원 latent space 를 학습시켜 다양체를 직접 그려봄 - 좌표를 옮기면 숫자 모양이 연속적으로 변함. latent space 가 실제로 채워져 있다는 시각적 증거
- 하한 최적화 속도에서 wake-sleep 알고리즘을 앞섬
- 주변 가능도 추정에서도 wake-sleep·Monte Carlo EM 을 상회

- latent 차원을 바꿔가며 무작위 샘플을 생성 → 학습 데이터에 없던 `z` 에서도 그럴듯한 숫자가 나옴
