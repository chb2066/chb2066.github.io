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

[Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)

## Abstract

**문제**
Siamese 구조의 SSL 은 붕괴(collapse)를 막기 위해 방법마다 장치를 하나씩 달고 있었음 - negative sample, 메모리 뱅크, 클러스터링 제약, 모멘텀 인코더
그런데 그중 무엇이 실제로 필요한지는 검증된 적이 없음

**해결책**
그 장치들을 전부 제거하고 무엇이 남아야 하는지 확인
stop-gradient 와 predictor 두 개만으로 붕괴가 막힌다는 것을 실험으로 보임
동작 원리를 두 변수의 교대 최적화(EM 유사)로 설명하는 가설 제시

---

## 1. Introduction

**붕괴라는 문제**
- 두 뷰의 표현을 가깝게 만들라고만 하면, 모든 입력에 대해 같은 값을 출력하는 것이 완벽한 해가 됨
- 그래서 방법마다 붕괴를 막는 장치를 하나씩 달고 있었음

| 방법 | 붕괴 방지 장치 |
|---|---|
| SimCLR | negative sample (큰 배치 필요) |
| MoCo | negative sample + 메모리 뱅크 + 모멘텀 인코더 |
| SwAV | 클러스터링 제약 (Sinkhorn 균등 배분) |
| BYOL | 모멘텀 인코더(EMA) + predictor + stop-gradient |

**이 논문의 질문** - 그렇다면 정말로 필요한 것은 무엇인가. 하나씩 제거해보면 무엇이 남는가.

---

## 3. Method

SimSiam 은 BYOL 과 비슷하게 stop-gradient 를 쓰는 구조지만, 두 branch 의 파라미터가 완전히 동일하다. 사실상 EMA만 제거한 것이다.

**남은 것은 두 개뿐**
- **predictor** - 한쪽 branch 에만 붙는 MLP
- **stop-gradient** - 반대쪽 branch 로는 기울기를 보내지 않음

negative sample 도, 메모리 뱅크도, 클러스터링도, 모멘텀 인코더도 없다.

**입력 요소**
- 같은 이미지의 두 augmented view
- 두 view 가 같은 encoder `f`(backbone + projection MLP)를 통과
- 한쪽만 predictor `h` 를 추가로 통과

**Loss - negative cosine similarity**
```text
D(p, z) = -(p/||p||₂) · (z/||z||₂)

p: predictor 를 거친 브랜치 출력
z: predictor 없이 나온 브랜치 출력 (stop-gradient 적용)
```

양쪽 브랜치를 서로 바꿔서(symmetrized) 두 번 계산한 뒤 평균낸다.
```text
L = 1/2 · D(p₁, stopgrad(z₂)) + 1/2 · D(p₂, stopgrad(z₁))
```

---

## 4. Empirical Study

### 4.1 Stop-gradient

논문의 핵심 실험이다. stop-gradient 를 제거하면 즉시 붕괴한다.

- 손실이 가능한 최소값(−1)로 곧장 떨어짐 → 모든 출력이 같아졌다는 뜻
- 표현의 표준편차가 0에 가까워짐. 정상 학습에서는 `1/√d` 근처를 유지함

이게 중요한 이유는 stop-gradient 가 최적화를 돕는 보조 장치가 아니라 붕괴를 막는 필수 요소라는 것을 보여주기 때문이다. 빼면 성능이 조금 나빠지는 게 아니라 학습 자체가 무너진다.

### 4.2 Predictor

- **predictor 를 제거하면 역시 붕괴** - 두 branch 가 완전히 대칭이 되면 자명한 해로 수렴함
- predictor 를 고정된 랜덤 초기화로 두면 학습이 수렴하지 않음 → 학습되는 predictor 여야 함
- predictor 의 학습률을 감쇠시키지 않는 편이 오히려 나음

### 4.3 Batch Size

- 배치 64 ~ 4096 전 구간에서 안정적으로 학습
- 배치 크기 의존이 낮음 - SimCLR 처럼 큰 배치가 필요하지 않음. negative sample 을 쓰지 않기 때문

### 4.4 Batch Normalization

- MLP head 의 BN 을 전부 제거하면 성능이 크게 떨어지지만 붕괴하지는 않음
- 즉 BN 은 최적화를 돕는 장치일 뿐, 붕괴 방지의 요소가 아님

### 4.5 Similarity Function · 4.6 Symmetrization

- cosine similarity 를 cross-entropy 형태로 바꿔도 동작 → 붕괴 방지가 특정 loss 형태에 의존하지 않음
- 대칭화는 성능을 올리지만 붕괴 방지와는 무관 - 비대칭 버전도 무너지지 않음

### 4.7 Summary

- 성능을 올리는 요소(BN, 대칭화, loss 형태)와 붕괴를 막는 요소(stop-gradient, predictor)는 별개
- 후자 둘만 남기면 충분함

---

## 5. Hypothesis

논문이 제시하는 가설은 이 구조를 두 변수를 번갈아 최적화하는 문제로 보는 것이다.

**두 변수**
- 하나는 네트워크 파라미터
- 다른 하나는 각 이미지의 표현(일종의 잠재 변수)

이렇게 보면 stop-gradient 는 자연스럽다. 한쪽을 고정한 채 다른 쪽을 최적화하는 EM 알고리즘의 구조와 같아지기 때문이다. predictor 는 그 최적화 과정에서 기대값을 근사하는 역할을 한다.

- **5.2 Proof of concept** - 교대 최적화를 명시적으로 구현해도 비슷하게 동작함을 확인
- **5.3 Discussion** - 가설이고 완전한 증명은 아니지만, 실험 결과와 일관된 설명을 줌

**남는 의문 하나**

> 기존 contrastive 방법들은 target 을 거의 업데이트하지 않으면서, 붕괴 방지를 위해 EMA 로 조금씩 업데이트한다. 하지만 gradient 를 그대로 사용하면 학습 업데이트가 너무 빨라지지 않나?
>
> → 업데이트가 급격하지 않다면 학습에 큰 영향을 주지 않고, target 을 stop-gradient 하는 것만으로도 충분하다.

논문의 답은 EMA 가 성능을 조금 올려주기는 하지만 붕괴 방지에는 필요하지 않다는 것이다. BYOL 에서 EMA 가 필수처럼 보였던 것은, 실제로는 stop-gradient 가 하던 일을 EMA 덕분으로 오해한 면이 있다.

---

## 6. Comparisons

- 6.1 ImageNet linear evaluation 에서 경쟁력 있는 성능. 특히 100 epoch 같은 짧은 학습에서는 다른 방법들을 앞섬
- 전이 학습(검출·분할)에서도 경쟁력 유지
- 6.2 SimCLR·SwAV·BYOL 을 SimSiam 의 특수한 경우로 배치해 각 방법이 무엇을 더 얹었는지 정리

구조가 가장 단순한데 성능이 크게 뒤지지 않는다는 것이 요지다.

---

## 정리

SimSiam의 값어치는 새 방법을 제안한 게 아니라 무엇이 불필요했는지 밝힌 것에 있다.

negative sample, 메모리 뱅크, 클러스터링, 모멘텀 인코더 - 이들이 전부 붕괴 방지를 위해 도입됐다고 여겨졌는데, 하나씩 빼보니 stop-gradient와 predictor만 있으면 됐다.

여기서 옮겨갈 만한 것은 방법이 아니라 태도다. 여러 장치가 함께 쓰이고 있을 때, 각각이 정말 필요한지는 따로 확인해야 안다. 관행적으로 함께 쓰이던 것들이 사실은 한 가지 이유를 중복해서 다루고 있을 수 있다.

다만 한계도 분명하다. 왜 되는지에 대한 이론이 완전하지 않고, 붕괴 방지가 미묘한 균형에 의존해서 하이퍼파라미터에 민감하다.

