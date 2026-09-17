---
title: InstructGPT
paper: Training language models to follow instructions with human feedback
venue: NeurIPS 2022
authors: Long Ouyang, Jeff Wu, Xu Jiang, et al.
link: https://arxiv.org/abs/2203.02155
claim: 모델을 키우는 것만으로는 사용자 의도를 따르지 않는다. 사람 선호로 정렬하면 100배 작은 모델이 더 선호된다.
tags: [RLHF, LLM, Alignment]
tier: basic
date: 2025-04-16
draft: true
---

> 📄 [**Training language models to follow instructions with human feedback**](https://arxiv.org/abs/2203.02155) · NeurIPS 2022 · Ouyang, Wu, Jiang et al.

## Abstract

**문제**
**대규모 언어 모델은 크기를 키운다고 사용자의 의도를 더 잘 따르지 않음**
허위 정보 생성, 유해한 내용 생성, 편향된 출력, 단순히 지시를 따르지 않음 같은 행동이 자주 나타남

**해결책**
**SFT → 보상 모델 → PPO** 의 3단계로 사람의 선호를 최적화 목표로 삼음
사전학습 분포의 로그 가능도를 함께 올리는 **PPO-ptx** 로 alignment tax 를 완화
파라미터가 **100배 적은** 1.3B 모델이 175B GPT-3 보다 사람에게 더 선호됨

---

## 1. Introduction

![Figure 1](/img/instructgpt/f1.png)

**정렬(alignment)의 목적** - 언어 모델이 사용자의 **명시적 의도**(지시 따르기)뿐만 아니라, 진실성·비편향성·무해성 같은 **암묵적 의도**에 따라 행동하도록 훈련하는 것. 논문은 모델이 **유용하고(Helpful), 정직하고(Honest), 무해해야(Harmless)** 한다고 명시한다.

**본 논문의 기여**
1. 사람 선호 기반 정렬 절차를 대규모 API 프롬프트 분포에서 실증
1. **모델 크기보다 정렬이 선호도에 더 크게 기여**함을 보임
1. **alignment tax** 를 측정하고 PPO-ptx 로 완화하는 방법을 제시

---

## 3. Methods and experimental details

### 3.1 High-level methodology

![Figure 2](/img/instructgpt/f2.png)

![그림](/img/rlhf/04.png)

**1단계. 지도 파인튜닝(SFT)**
1. prompt dataset 을 가져옴
1. 라벨러가 각 prompt 에 대한 **좋은 예시 답변**을 직접 작성함
1. 이것으로 GPT-3 를 지도학습 파인튜닝해서 예시 답변과 비슷한 답변이 나오게 함

여기서 학습되는 것은 **질문에서 답변으로의 매핑**이다. 좋은 답변 텍스트의 전체 토큰을 사용해 모델의 답변과 토큰 단위로 예측을 비교하고, 토큰 단위 loss 로 학습한다.

**2단계. 보상 모델 학습**
1. 프롬프트와 그에 대한 여러 모델의 출력을 가져옴

   > *"most of our comparison data comes from our supervised policies, with some coming from our PPO policies"*

   대부분은 1단계 모델의 출력을 쓰고, 일부는 PPO 로 정책을 갱신한 모델의 출력을 가져온다.

1. 이것을 사람의 판단으로 **순위를 매겨 비교쌍**으로 만듦 → 인간의 선호를 학습할 수 있는 데이터가 됨

   예를 들어 `B > A > C > D` 라면 `B>A`, `B>C`, `B>D`, `A>C`, `A>D` 같은 쌍으로 분해한다. 선호되는 응답이 더 높은 점수를 받는 구조다.

1. 이 데이터로 보상 모델을 학습함

```python
loss = -E[log(σ(r_θ(x, y_w) - r_θ(x, y_l)))]

# r_θ(x, y_w): 선호된 응답의 보상
# r_θ(x, y_l): 덜 선호된 응답의 보상
# σ: sigmoid 함수
```

**3단계. PPO 로 정책 최적화** - 새 프롬프트를 가져와서 보상 모델의 점수를 보상으로 삼아 정책을 갱신한다.

### 3.2 Dataset · 3.3 Tasks

**입력 요소**
- OpenAI API 에 실제로 제출된 프롬프트 분포
- 생성, 열린 QA, 브레인스토밍, 대화, 요약, 분류, 추출 등 다양한 사용 사례
- 라벨러가 직접 작성한 프롬프트로 초기 분포를 보충

### 3.4 Human data collection

- 라벨러를 선별 시험으로 뽑고, **연구자와 기준을 맞춰가며** 작업
- 학습에 참여하지 않은 별도 라벨러 집단으로도 평가해 **기준 과적합 여부**를 확인

### 3.5 Models - PPO-ptx

**문제** - 보상 모델만 최적화하면 **공개 NLP 벤치마크 성능이 떨어진다.** 정렬을 얻는 대가로 기존 능력을 잃는 것이고, 논문은 이를 **alignment tax** 라고 부른다.

**해결** - PPO 에 **사전 훈련 분포의 로그 가능도를 높이는 업데이트를 혼합**한다. 정렬 목표를 좇으면서도 원래 언어 모델링 능력을 붙잡아두는 것이다.

이 형태는 익숙하다. "**향상 도중 잃는 축을 방어한다**"는 점에서 WiSE-FT 나 KL 정박과 목적이 같다. 구현하는 공간만 다르다.

---

## 4. Results

### 4.1 Results on the API distribution

![Figure 3](/img/instructgpt/f3.png)

- **GPT-3 보다 파라미터가 100배 적은데도** 인간 선호가 더 높음
- 학습에 참여하지 않은 라벨러가 평가해도 결과가 유지됨 → 특정 라벨러 취향에 과적합된 것이 아님

![Figure 4](/img/instructgpt/f4.png)

- 지시 준수, 제약 조건 준수, 할루시네이션 항목 모두에서 개선
- 할루시네이션이 **절반으로** 줄었음
- respectful prompt 가 주어졌을 때 GPT-3 보다 **유해한 출력이 25% 감소**

### 4.2 Results on public NLP datasets

![Figure 6](/img/instructgpt/f6.png)

- TruthfulQA 에서 진실성이 개선
- PPO 만 쓰면 공개 벤치마크가 하락하지만 **PPO-ptx 는 그 하락을 대부분 회복**

---

## 한계

**누구에게 정렬되었는가**의 문제가 남는다. OpenAI API 고객의 선호에 정렬된 것이므로, 넓은 범위의 '사람'의 선호가 아니라 **특정 계층의 선호**를 따른다.

그리고 유해성과 무관한 생성이라 **유해한 정보를 생성할 수 있다.** 심지어 모델이 특정 사용자 지시를 따르지 않을 수도 있다.

**정렬 연구가 필요한 이유** - model pretraining 에 비해 정렬을 위한 훈련 비용이 상대적으로 적다. 그렇다면 **큰 모델을 새로 훈련하는 것보다 기존 모델을 정렬하는 데 힘쓰는 게 더 효율적일 수 있다.** 이 관찰이 이후 방향을 상당 부분 결정했고, 100배 작은 모델이 정렬만으로 더 선호된다는 결과가 그 근거다.

---

## 정리

InstructGPT가 확립한 것은 **SFT → 보상 모델 → PPO**라는 3단계 절차다. 이후 거의 모든 대화형 LLM이 이 골격을 따른다.

그리고 함께 드러난 것이 **alignment tax**다. 한 방향으로 최적화하면 다른 축을 잃는다는 것이고, 그래서 PPO-ptx처럼 원래 분포를 붙잡는 항이 필요해진다. 이 긴장은 이 계열 전체에서 반복된다.

---

*`f` 로 시작하는 그림은 원 논문에서 가져왔다. Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), NeurIPS 2022.*
