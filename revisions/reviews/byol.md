<!--
개정: 2026-09-10 (원본: src/content/reviews/byol.md)
- 원본이 25편 중 가장 짧았다(883자). 논문에서 보강이 가장 많이 필요했다
- 원본에서 teacher / online 용어가 뒤섞여 있었다. 논문 용어인
  online network / target network 로 통일하고, 어느 쪽에 predictor 가 붙는지를 명확히 했다
  (원본: "teacher는 predictor가 없고 Projection한 logit을 online으로 보냄" — 방향이 모호)
- 「왜 붕괴하지 않는가」를 논문 근거로 확장 — 원본은 "EMA 로 천천히 업데이트하기 때문"
  한 줄이었다. 실제로는 predictor + stop-gradient 의 비대칭이 함께 작용하고,
  논문은 이를 "target 이 online 을 따라오는 부트스트랩"으로 설명한다
- 「구조」에 projection head 와 predictor 의 위치를 명시. 원본의 서술만으로는
  어느 지점에서 loss 를 계산하는지 알기 어려웠다
- 「실험에서 확인된 것」 신설 — ImageNet linear 74.3%(ResNet-50), 배치 크기와
  augmentation 선택에 대한 강건성. 원본에 결과가 없었다
- 끝맺음을 평서형으로 통일
-->
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

핵심 키워드:
negative 없는 자기지도, online/target 비대칭, predictor, EMA 부트스트랩
주요 전략:
음성 샘플 대신 한쪽에만 predictor를 두는 비대칭과 stop-gradient로 붕괴를 막는다
사용 가능 분야:
음성 샘플을 정의하기 어려운 도메인의 표현 학습. 의료 영상, 로봇 궤적 등

#### 배경 지식

기존 contrastive learning — MoCo, SimCLR 등 — 은 **negative pair와 positive pair를 모두 사용**해서 학습했다. 같은 것은 가깝게, 다른 것은 멀게 만드는 방식이다.

여기서 음성 샘플이 하는 일은 붕괴 방지다. 음성이 없으면 모든 표현을 같은 값으로 만드는 자명한 해가 생긴다.

**BYOL은 positive pair만으로 학습한다.** 음성 샘플이 전혀 없다.

이게 왜 중요한가. **음성을 정의하기 어려운 도메인**이 많기 때문이다. 의료 영상에서 두 환자의 스캔이 정말 "다른 것"인지, 로봇 궤적에서 두 시점이 정말 무관한지는 자명하지 않다. 음성 샘플에 의존하지 않는 방법은 그런 곳에서 결정적인 이점을 갖는다.

#### BYOL method

**두 네트워크**

- **online network** — 학습 대상이다. encoder → projection head → **predictor**로 이어진다.
- **target network** — encoder → projection head까지만 있고 **predictor가 없다.**

동일한 이미지를 다르게 augmentation한 두 view에 대해 학습한다.

**흐름**

1. 같은 이미지에서 두 개의 view를 만든다.
2. view 1은 online network를, view 2는 target network를 통과한다.
3. online network는 projection 뒤에 **predictor(MLP)**를 한 번 더 거친다.
4. predictor의 최종 출력 차원이 target network의 projection 출력 차원과 같으므로, **두 값을 직접 비교**할 수 있다.
5. 그 차이를 loss로 줄인다.

**비대칭이 핵심이다.** online 쪽에만 predictor가 있고, target 쪽에는 없다. 양쪽이 대칭이면 두 출력이 같아지는 자명한 해로 무너진다.

#### Loss와 갱신

**Loss**
MSE를 쓴다. 정확히는 정규화된 두 벡터의 평균제곱오차이고, 이는 cosine similarity와 동등하다.

**갱신 방식**

- **online network** — loss로 역전파해서 학습한다. target의 분포와 최대한 가까워지는 방향으로 움직인다.
- **target network** — 역전파하지 않는다. **online network의 파라미터를 EMA로 따라간다.**

```text
ξ ← τ·ξ + (1 − τ)·θ

ξ: target 파라미터
θ: online 파라미터
τ: 모멘텀 계수 (1에 가까움)
```

#### 왜 붕괴하지 않는가

음성 샘플 없이 "두 출력을 가깝게 만들라"고만 하면 모든 것을 같은 값으로 내는 해가 존재한다. BYOL이 그리로 가지 않는 이유는 두 가지가 함께 작용하기 때문이다.

**1. target이 천천히 변한다**
target은 EMA로 갱신되므로 큰 영향을 주지 않고 조금씩만 변한다. online이 target을 쫓아가는 동안 target은 거의 고정된 목표로 남는다. 그래서 **다양한 정보를 계속 사용할 수 있다.**

**2. predictor와 stop-gradient의 비대칭**
online에만 predictor가 있고 target으로는 기울기가 흐르지 않는다. 이 비대칭 때문에 **두 네트워크가 같은 자명한 해로 동시에 수렴하지 못한다.**

**부트스트랩이라는 이름의 의미**
target network는 online network의 과거 버전이다. 즉 **자기 자신의 이전 상태를 목표로 삼아 스스로를 끌어올린다.** 외부 정답이 없는데도 학습이 진행되는 이유가 여기 있고, "Bootstrap Your Own Latent"라는 제목이 그것을 가리킨다.

#### 실험에서 확인된 것

- **ImageNet linear evaluation 74.3%** (ResNet-50). 음성 샘플을 쓰는 당시 최고 방법들을 앞섰다.
- **배치 크기에 덜 민감하다.** SimCLR은 배치가 작아지면 성능이 크게 떨어지는데, BYOL은 음성 샘플에 의존하지 않으므로 그 영향이 작다.
- **augmentation 선택에도 더 강건하다.** contrastive 방법은 특정 augmentation(색상 왜곡 등)을 빼면 성능이 급락하는데, BYOL은 덜 그렇다.

두 번째와 세 번째가 실용적으로 중요하다. 큰 배치를 감당할 수 없거나, 도메인에 맞는 augmentation을 잘 모를 때 선택지가 된다.

#### 정리

BYOL이 보여준 것은 **음성 샘플이 붕괴 방지의 유일한 방법은 아니라는 것**이다.

같은 것을 가깝게 만들되 무너지지 않게 하려면 어떤 형태의 비대칭이 필요한데, 그 비대칭을 **음성 샘플로 만들 수도 있고 구조로 만들 수도 있다.** BYOL은 후자를 택했다.

그리고 여기서 떼어 쓸 수 있는 부품이 두 개 나온다.

**첫째, EMA target.** "느리게 변하는 목표"라는 장치는 자기지도뿐 아니라 강화학습의 타깃 네트워크, 지식 증류의 교사 등 어디서나 재사용된다.

**둘째, predictor + stop-gradient 조합.** 이후 SimSiam이 여기서 EMA까지 빼도 된다는 것을 보인다. 즉 BYOL이 필수라고 여겼던 것 중 일부는 사실 없어도 됐다.

다만 대가가 있다. **붕괴 방지가 미묘한 균형에 의존해서 하이퍼파라미터에 민감하고, 왜 동작하는지에 대한 이론이 완전하지 않다.**
