<!--
개정: 2026-09-10 (원본: src/content/reviews/dit.md)
- 원본의 "fc없이 출" 처럼 끊긴 문장 완성
- 「조건 주입 방식 비교」 신설 — 논문이 실제로 비교한 네 가지(in-context, cross-attention,
  adaLN, adaLN-Zero)와 각각의 Gflops 비용. 원본은 adaLN-Zero만 설명해서
  "왜 그걸 골랐는가"가 빠져 있었다
- 「스케일링」 신설 — 이 논문의 제목이 Scalable 인 이유. 모델 4종 구성표(Table 1),
  patch size가 토큰 수를 정하는 구조, Gflops와 FID의 관계
- 결과 수치 보강 — ImageNet 256×256 FID 2.27
- adaLN-Zero 의 zero-init 근거(ResNet의 identity 초기화 계보) 보강
- 끝맺음을 평서형으로 통일
- 원본의 adaLN-Zero 메커니즘 서술과 CFG 관찰은 그대로 유지
-->
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

핵심 키워드:
diffusion 백본을 U-Net에서 Transformer로 교체, adaLN-Zero 조건 주입, 예측 가능한 스케일링
주요 전략:
latent 공간에서 동작하는 Transformer를 만들고, 조건 주입 방식 네 가지를 비교해 adaLN-Zero를 고른다
사용 가능 분야:
클래스 조건부 이미지 생성. 이후 대부분의 대규모 diffusion 모델이 이 백본을 따른다

#### 배경 지식

**기존 생성 방식**
- **DDPM** — 노이즈를 점진적으로 더하고 그 역과정을 복원하며 이미지를 생성한다.
- **VAE** — latent space로 압축했다가 복원하며 이미지를 생성한다. 평균과 표준편차를 다룬다.

**이 논문의 질문**
diffusion 모델의 백본은 관행적으로 U-Net이었다. 그런데 Transformer는 다른 분야에서 **뛰어난 스케일링 특성**을 보여줬다. 그래서 묻는다 — diffusion에서 U-Net을 Transformer로 바꾸면 어떻게 되는가. 그리고 그 스케일링 특성이 따라오는가.

#### 전체 흐름

1. 원본 이미지를 입력한다.
2. 사전 학습된 **VAE Encoder**를 통과시킨다. 기존에는 fc를 거쳤다면 여기서는 fc 없이 출력한다.
3. **latent `z`**를 뽑는다.
4. `z`를 **DiT**에 입력한다.

즉 DiT는 픽셀이 아니라 latent 공간에서 동작한다. 이 부분은 Latent Diffusion과 같다.

#### DiT method

**Patchify**
- 입력은 latent image다.
- 패치화해서 일렬로 나열한다.
- positional embedding으로 위치 정보를 더한다.

여기서 **patch size `p`가 설계 다이얼**이다. `p`가 작을수록 토큰 시퀀스가 길어지고 Gflops가 늘어난다. 논문은 `p ∈ {2, 4, 8}`을 비교하는데, **작은 패치가 일관되게 더 낮은 FID를 낸다.**

**DiT Block**

adaLN-Zero(Adaptive Layer Norm - Zero, scale 파라미터를 0으로 초기화)를 쓴다.

*기존 방식*
- Layer norm은 γ, β로 한 이미지 내의 채널을 정규화한다.
- 시간 `t`와 클래스 label `c`는 단순히 더하는 방식으로 값의 분포를 바꿨다.

*adaLN-Zero 방식*
- **γ, β가 `t`, `c`에 의해 바뀌도록** 하는 layer norm을 쓴다. 단순 값 변환이 아니라 feature map의 강도와 분포를 결정하게 만드는 것이다.
- 메커니즘:
  - `z = Emb(t) + Emb(c)`
  - `MLP(z) = γ, β`
  - `adaLN(x, z) = γ(z) · LayerNorm(x) + β(z)`

![그림 1](/img/dit/01.png)

*블록 구조*
- adaLN-Zero를 통과하며 시간과 클래스 정보가 주입된다.
- **Self-Attention**으로 전역 정보를 얻고 noise 부분을 강조해 학습한다.
- **Pointwise MLP**(각 패치에 대한 MLP)로 정보를 가공하고 업데이트한다.

**Final Layer**
- Standard Layer Norm과 Linear로 데이터를 정리하고 차원을 맞춘다.
- unpatchify로 재배치한다.
- conv로 최종 출력을 낸다. VAE latent `z`의 예상 noise다.

#### 조건 주입 방식 비교

논문이 실제로 비교한 것은 **네 가지**다. adaLN-Zero가 왜 선택됐는지는 이 비교를 봐야 안다.

| 방식 | 어떻게 | Gflops 비용 |
|---|---|---|
| **In-context** | `t`와 `c`의 임베딩을 추가 토큰 두 개로 시퀀스에 붙인다. ViT의 cls token과 비슷하다. 마지막 블록 뒤에 제거한다 | 거의 없음 |
| **Cross-attention** | `t`, `c`를 길이 2의 별도 시퀀스로 두고, self-attention 뒤에 cross-attention 층을 추가한다 | **가장 큼, 약 15%** |
| **adaLN** | γ, β를 직접 학습하지 않고 `t`와 `c` 임베딩의 합에서 회귀한다 | 가장 적음 |
| **adaLN-Zero** | adaLN에 더해 **블록을 항등함수로 초기화**한다 | 거의 없음 |

**adaLN-Zero가 학습의 모든 단계에서 나머지 셋을 앞선다.** cross-attention은 가장 비싼데 성능은 더 낮다.

**adaLN-Zero의 zero-init이 왜 좋은가**
ResNet 계열에서 각 residual block을 항등함수로 초기화하는 게 이롭다는 것이 알려져 있었다. 각 블록의 마지막 batch norm scale을 0으로 초기화하면 대규모 학습이 빨라진다는 관찰이 있고, diffusion U-Net도 residual 연결 직전의 마지막 conv를 zero-init한다.

DiT는 같은 것을 한다. **MLP가 모든 스케일 파라미터에 대해 zero-vector를 출력하도록 초기화해서, DiT 블록 전체가 처음에 항등함수가 되게** 만든다.

**adaLN 계열의 제약 하나** — 세 가지 블록 설계 중 adaLN만이 **모든 토큰에 같은 함수를 적용하도록 제한**된다. 조건이 공간적으로 균일하게 작용한다는 뜻이다. 클래스 label처럼 전역적인 조건에는 맞지만, 위치마다 다른 조건을 줘야 한다면 이 방식은 부적합하다.

#### 스케일링

이 논문의 제목이 *Scalable*인 이유가 여기 있다.

**모델 구성** — ViT의 설정을 따라 층 수, hidden size, head 수를 함께 키운다.

| 모델 | Layers | Hidden size | Heads | Gflops (I=32, p=4) |
|---|---|---|---|---|
| DiT-S | 12 | 384 | 6 | 1.4 |
| DiT-B | 12 | 768 | 12 | 5.6 |
| DiT-L | 24 | 1024 | 16 | 19.7 |
| DiT-XL | 28 | 1152 | 16 | 29.1 |

patch size와 조합하면 **0.3에서 118.6 Gflops까지** 커버한다.

**핵심 관찰** — 모델 Gflops가 늘면 FID가 꾸준히 내려간다. 모델을 키우는 것과 패치를 줄이는 것 **양쪽 모두** 효과가 있고, 둘 다 결국 "토큰당 연산량을 늘리는" 같은 방향이다.

**결과** — 가장 큰 DiT-XL/2는 기존 U-Net 기반 diffusion 모델(ADM, LDM)을 전부 앞서면서 연산 효율도 좋다. ImageNet 256×256 클래스 조건부 생성에서 **FID 2.27**로 당시 최고 성능이다.

#### 학습과 생성

- **학습** — 실제 노이즈와 DiT의 예측 noise 차이로 학습한다.
- **생성** — DiT가 낸 noise를 latent `z`에서 빼고 VAE decoder에 넣어 이미지를 출력한다.

**Classifier-Free Guidance**
DiT 실험에서 생성 품질을 크게 끌어올리는 요소로 CFG를 쓴다. 조건(class label `c`)이 있는 예측과 없는 예측을 각각 구해서, 그 차이를 증폭하는 방향으로 노이즈 예측을 보정한다. 조건에 더 충실하면서도 품질 높은 샘플이 나온다.

#### 정리

이 논문이 남긴 것은 **"diffusion 백본에 Transformer의 스케일링 법칙이 그대로 적용된다"**는 확인이다. U-Net의 귀납 편향이 없어도 되고, 오히려 없는 편이 크게 키울 때 유리하다.

설계 측면에서 가져갈 것은 **adaLN-Zero**다. 전역 조건을 아주 싸게 주입하면서, zero-init으로 학습 초기를 안정화한다. 이 조합은 diffusion 밖에서도 동결 백본에 조건을 붙일 때 반복해서 등장한다.
