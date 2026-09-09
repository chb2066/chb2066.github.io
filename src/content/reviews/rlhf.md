---
title: 강화 학습과 RLHF
claim: PPO부터 InstructGPT·WebGPT까지, 인간 선호도를 보상 모델로 바꿔 정책을 최적화하는 흐름.
tags: [RL, RLHF, LLM]
tier: basic
date: 2025-04-16
draft: true
---

#### POLICY란?

- Self-Attention 가중치 (Q, K, V 행렬)
  - q, k, v 의 가중치 업데이트
- Feed-Forward 가중치
  - FFN network에서 GELU 같이 비선형 함수로 인해 없어지는 값을 조절할 수 있음
    - 즉, ex 가중치를 더하거나 빼서 특정 성질에 대한 벡터값이 변하면 backward를 통해서 중요도에 따른 제외, 강조 등이 가능해짐.
- Layer Normalization 파라미터
  - 각 feature에 대한 norm을 사용해서 위와 같이 특징별로 중요도에 따른 제외, 강조 등 가능해짐.
- Positional Embedding
  - 0, 1, 2, 3 등 토큰 단위로 positional embedding 되어 있다고 가정. 각 토큰에 대한 임베딩이 더 관련있는 토큰과 가까워지게 함. 의미론 증대

```python
# 보상을 최대화하는 방향으로
gradient = ∇_θ (보상 - KL페널티)

# 각 파라미터를 조금씩 수정
새_가중치 = 기존_가중치 + learning_rate * gradient

# 너무 큰 변화 방지
if 변화량 > clip_범위:
    변화량 = clip_범위
```

#### PPO 알고리즘

[https://arxiv.org/pdf/1707.06347](https://arxiv.org/pdf/1707.06347)
1. N 개의 병렬 액터로 데이터 수집
1. 수집된 데이터로 일정 에폭동안 minibatch 학습으로 정책 최적화
1. 정책 업데이트
1. CLIP 을 통해서 정책 변화가 너무 커지지 않게 막음

![그림 1](/img/rlhf/01.png)

용례:

![그림 2](/img/rlhf/02.png)

1. 병렬 액터가 환경을 동시에 실행
  1. 각 환경은 독립적으로 에이전트와 상호작용
1. 현재 상태(St), Reward(Rt)를 에이전트에게 전달
1. Agent 가 PRO algorithm을  실행.
  1. 병렬 액터들에게 전달받은 데이터 수집
  1. 수집된 데이터를 일정 에폭동안 minibatch 학습으로 정책 최적화
  1. 정책 업데이트
1. 업데이트 된 정책을 사용한 행동 결정(At)
  1. 모든 환경에서 같은 정책을 사용하지만 다른 행동 가능
1. 반복

#### Learning to summarize from human feedback(RLHF 관련 첫 논문)

[https://arxiv.org/pdf/2009.01325](https://arxiv.org/pdf/2009.01325)

![그림 3](/img/rlhf/03.png)

1. 데이터 수집(Redit 같은 곳에서 원 데이터 가져옴.).
1. 수집 데이터 기반으로 미리 만들어진 정책을 주고 병렬적 액터(한 정책에 한 액)가 요약문 생성.
  1. 정책은 아래와 같은 것들로 변화를 만든다.
    1. 다른 하이퍼파라미터
    1. 다른 랜덤 시드
    1. 다른 훈련 에폭
    1. 베이스라인 모델
1. 인간 평가자가 더 좋은 걸 선택하고 이를 통한 비교 데이터 수집
1. 요약문 집합을 리워드 모델에 집어 넣어서 리워드 뽑아서 비교 데이터(인간 평가자가 더 좋은 걸 선택)를 기반으로 loss 구함.(인간 선호도 예측 모델 만들기)
1. 이 모델을 사용해 PPO 진행
  1. 1의 데이터와 다른 새 데이터 수집
  1. 새 데이터 및 현재 정책으로 요약을 생성. 보상 모델이 점수 부여, 이를 통해 정책 최적화(일정 에폭 mini-batch 학습을 통해 정책 최적화(인간 평가자의 기준으로 정책이 최적화됨.))
  1. 정책 업데이트(Clipping으로 너무 큰 업데이트는 안됨)

#### Training language models to follow instructions with human feedback

문제 인식:
대규모 언어 모델(LM)은 단순히 크기를 키운다고 해서 사용자의 의도를 더 잘 따르지는 않습니다. 실제로는 허위 정보 생성, 유해한 내용 생성, 편향된 출력, 또는 단순히 사용자 지시를 따르지 않는 등 의도하지 않은 행동을 자주 보입니다.
정렬(alignment)의 목적: 
언어 모델이 사용자의 명시적 의도(지시 따르기)뿐만 아니라, 진실성, 비편향성, 무해성과 같은 암묵적 의도에 따라 행동하도록 훈련하는 것. 모델은 유용하고(Helpful), 정직하고(Honest), 무해해야(Harmless) 한다고 명시됨.

인간 선호도 기반 finetuning

PPO-ptx (Pretraining mix)의 도입:
PPO에 **사전 훈련 분포의 로그 가능도를 높이는 업데이트를 혼합**하여

결과: GPT-3보다 매개변수 100배 감소, **인간 선호도 증가**, 할루시네이션 절반으로 감소, respectful prompt 가 주어졌을 때 GPT-3 보다 25% 유해한 출력 감소

한계: 누구에게 정렬되었는 가 에 따른 문제 발생 가능, OPEN API 고객의 선호도에 정렬된 것이기 때문에 넓은 범위의 ‘사람’의 선호가 아닌 특정 계층 선호에 따른다.

유해성 무관 생성임으로 유해한 정보 생성 가능하다.  심지어 모델이 특정 사용자 지시를 안따를 수도 있다는 특징이 존재한다.

함의: 정렬 연구의 필요성: model pretraining 에 비해 align을 위한 훈련 비용이 상대적으로 적다. 큰 모델 훈련보다 기존 모델 정렬에 힘쓰는 게 더 효율적일 수 있다.

알고리즘:
[https://arxiv.org/pdf/2203.02155](https://arxiv.org/pdf/2203.02155)

![그림 4](/img/rlhf/04.png)

  1. prompt dataset 가져옴.
  1. 라벨러가 prompt에 대한 ‘좋은 예시 답변’을 작성한다.
  1. 이걸로 model(GPT-3)을 지도 학습으로 finetuning 해서 ‘좋은 예시 답변’과 비슷한 답변이 나오도록 한다.(**질문→답변의 매핑을 학습한다. 좋은 답변의 텍스트의 전체 토큰들을 사용해 GPT의 답변과 토큰 단위 예측 비교, 토큰 단위 loss를 사용하여 학습 진행**)
  1. 프롬프트와 프롬프트에 대한 몇몇 모델의 출력을 가져온다.
    1. “most of our comparison data comes from our supervised policies, with some coming from our PPO policies”
    1. 대부분은 step1의 모델 출력에서 나온 답변을 사용하고 몇몇개는 PPO를 통해 정책 업데이트한 모델의 출력을 가져옴.
  1. 이걸 사람의 판단 아래 랭킹을 맺어 순위 데이터, 비교쌍으로 만든다.(인간의 선호도를 학습할 수 있는 데이터)
    1. ex B>A>C>D 면 B>A, B>C, B>D, A>C, A>D 이런 식으로 상대적 선호도에 따른 쌍으로 만든 후 선호되는 응답이 더 높은 점수 받는 구조.

```python
loss = -E[log(σ(r_θ(x, y_w) - r_θ(x, y_l)))]

# r_θ(x, y_w): 선호된 응답의 보상
# r_θ(x, y_l): 덜 선호된 응답의 보상
# σ: sigmoid 함수
```

  1. 이걸로 리워드 모델 안에 넣어서 학습 진행함.
1. 새 프롬프트 가져와서 위 순위 데이터를 리워드로 넣어서 정책 업데이트 진행.

#### WebGPT: Browser-assisted question-answering with human feedback

**개괄:**
웹 브라우저에 검색하는 방식을 사용하여 긴 글에 대해 GPT-3가 답변하는 방식을 fine-tuning한다.
- human이 특정 질문에 답변하기 위해 “웹 브라우저에 접속하여 어떻게 행동하여 답변하는 가”에 대한 것을 학습하여 fine-tuning한다.
  - 행동은 text, token화하여 token 단위 loss로 학습 진행
- 이를 통해 사람의 행동을 모사하여 정보를 retrival하고 synthesis 하는 방식을 학습 시킬 것이다. 추론은 rejection sampling을 통해 여러 답변을 생성하고 reward model 기준으로 최고 점수 답변을 출력한다.

![그림 5](/img/rlhf/05.png)

![그림 6](/img/rlhf/06.png)

과정:
**1. 데이터 수집 및 학습 단계**
인간 사용자가 실제로 웹 브라우저를 사용하여 질문에 답하는 과정을 기록
- 사용자가 "How to train crows to bring me gifts"라는 질문에 답하기 위해 검색하고 정보를 찾는 모든 행동을 수집
- 이러한 데모 데이터를 사용하여 GPT-3를 fine-tuning (질문, [행동1, 행동2, ..., 최종답변] 시퀀스를 학습)

**2. WebGPT의 행동 명령어 체계**
Table 1 내용 확인

**3. 실제 작동 과정**
이미지 (b) 내용 확인

**4. 보상 모델과 rejection sampling**
- 인간이 선호하는 답변 패턴을 학습한 보상 모델 구축
- WebGPT가 생성한 여러 답변 중에서 보상 모델의 점수가 높은 것만 선택
