---
title: SimSiam
paper: Exploring Simple Siamese Representation Learning
venue: CVPR 2021
link: https://arxiv.org/abs/2011.10566
claim: EMA조차 없이 stop-gradient와 predictor만으로 collapse를 피할 수 있다.
tags: [Self-supervised, Non-contrastive]
tier: basic
date: 2025-06-20
draft: true
---

#### 주요 전략
1. negative pair, 메모리 뱅크, 클러스터링, 모멘텀 인코더를 전부 제거하고 무엇이 남아야 하는지 확인함.
2. stop-gradient와 predictor만으로 collapse 방지가 성립함을 확인함.

#### 배경 지식

자기지도 학습에서 **붕괴**(collapse)는 근본적인 문제다. 두 뷰의 표현을 가깝게 만들라고만 하면, 모든 입력에 대해 같은 값을 출력하는 것이 완벽한 해가 된다.

그래서 방법마다 붕괴를 막는 장치를 하나씩 달고 있었다.

| 방법 | 붕괴 방지 장치 |
|---|---|
| SimCLR | **negative sample** (큰 배치 필요) |
| MoCo | negative sample + **메모리 뱅크** + 모멘텀 인코더 |
| SwAV | **클러스터링 제약** (Sinkhorn 균등 배분) |
| BYOL | **모멘텀 인코더(EMA)** + predictor + stop-gradient |

**기존 방법의 흐름**

기존 contrastive learning에서는 teacher를 두고 EMA 같은 기법으로 갱신 속도를 조절했다. BYOL에서는 두 branch 중 하나에만 predictor를 두고 stop-gradient를 썼다.

**SimSiam의 질문**

그렇다면 **정말로 필요한 것은 무엇인가.** 하나씩 제거해보면 무엇이 남는가.

#### SimSiam method

SimSiam은 BYOL과 비슷하게 stop-gradient를 쓰는 구조지만, **target과 online의 파라미터가 동일하다.** 사실상 **EMA만 제거한 것**이다.

그러니까 남은 것은 두 개뿐이다.

- **predictor** — 한쪽 branch에만 붙는 MLP
- **stop-gradient** — 반대쪽 branch로는 기울기를 보내지 않음

negative sample도, 메모리 뱅크도, 클러스터링도, 모멘텀 인코더도 없다.

**Loss — negative cosine similarity**

```text
D(p, z) = -(p/||p||₂) · (z/||z||₂)

p: predictor 를 거친 online 브랜치 출력
z: predictor 없이 나온 target 브랜치 출력 (stop-gradient 적용)
```

양쪽 브랜치를 서로 바꿔서(symmetrized) 두 번 계산한 뒤 평균낸다.

```text
L = 1/2 · D(p₁, stopgrad(z₂)) + 1/2 · D(p₂, stopgrad(z₁))
```

#### stop-gradient를 제거했을 때

논문의 핵심 실험이다. **stop-gradient를 제거하면 즉시 붕괴한다.**

- 손실이 **가능한 최소값(-1)로 곧장 떨어짐.** 모든 출력이 같아졌다는 뜻임.
- 표현의 표준편차가 0에 가까워짐. 정상 학습에서는 `1/√d` 근처를 유지함.

이게 중요한 이유는 **stop-gradient가 최적화를 돕는 보조 장치가 아니라 붕괴를 막는 필수 요소**라는 것을 보여주기 때문이다. 빼면 성능이 조금 나빠지는 게 아니라 학습 자체가 무너진다.

**predictor도 필요하다.** predictor를 제거하면 역시 붕괴한다. 두 branch가 완전히 대칭이 되면 자명한 해로 수렴하기 때문이다.

#### 동작 원리 — EM 해석

논문이 제시하는 가설은 이 구조를 **두 변수를 번갈아 최적화하는 문제**로 보는 것이다.

- 한 변수는 네트워크 파라미터임.
- 다른 변수는 각 이미지의 표현(일종의 잠재 변수)임.

이렇게 보면 stop-gradient는 자연스럽다. 한쪽을 고정한 채 다른 쪽을 최적화하는 EM 알고리즘의 구조와 같아지기 때문이다. predictor는 그 최적화 과정에서 기대값을 근사하는 역할을 한다.

가설이고 완전한 증명은 아니지만, 실험 결과와 일관된 설명을 준다.

#### 생각해볼 만한 것

> 기존 contrastive 방법들은 target을 거의 업데이트하지 않으면서, 붕괴 방지를 위해 EMA로 조금씩 업데이트한다. 하지만 위처럼 gradient를 그대로 사용하면 학습 업데이트가 너무 빨라지지 않나?
>
> → 업데이트가 급격하지 않다면 학습에 큰 영향을 주지 않고, target을 stop-gradient하는 것만으로도 충분하다.

이 의문은 논문이 실제로 다루는 지점과 맞닿아 있다. 논문의 답은 **EMA가 성능을 조금 올려주기는 하지만 붕괴 방지에는 필요하지 않다**는 것이다. BYOL에서 EMA가 필수처럼 보였던 것은, 실제로는 stop-gradient가 하던 일을 EMA 덕분으로 오해한 면이 있다.

#### 실험에서 확인된 것

- **ImageNet linear evaluation에서 경쟁력 있는 성능**을 냄. 특히 **100 epoch 같은 짧은 학습에서는 다른 방법들을 앞섬.**
- **배치 크기 의존이 낮음.** SimCLR처럼 큰 배치가 필요하지 않음. negative sample을 쓰지 않기 때문임.
- 구조가 가장 단순한데 성능이 크게 뒤지지 않는다는 것이 요지임.

#### 정리

SimSiam의 값어치는 새 방법을 제안한 게 아니라 **무엇이 불필요했는지 밝힌 것**에 있다.

negative sample, 메모리 뱅크, 클러스터링, 모멘텀 인코더 — 이들이 전부 붕괴 방지를 위해 도입됐다고 여겨졌는데, 하나씩 빼보니 **stop-gradient와 predictor만 있으면 됐다.**

여기서 옮겨갈 만한 것은 방법이 아니라 **태도**다. 여러 장치가 함께 쓰이고 있을 때, 각각이 정말 필요한지는 따로 확인해야 안다. 관행적으로 함께 쓰이던 것들이 사실은 한 가지 이유를 중복해서 다루고 있을 수 있다.

다만 한계도 분명하다. **왜 되는지에 대한 이론이 완전하지 않고**, 붕괴 방지가 미묘한 균형에 의존해서 하이퍼파라미터에 민감하다.
