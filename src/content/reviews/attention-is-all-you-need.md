---
title: Attention Is All You Need
paper: Attention Is All You Need
venue: NeurIPS 2017
link: https://arxiv.org/abs/1706.03762
claim: recurrence 없이 attention만으로 시퀀스를 병렬 처리하면서도 장거리 의존성을 잡을 수 있다.
tags: [Transformer, NLP]
tier: basic
date: 2025-03-24
draft: true
---

## **사전 개념(attention)**

![그림 1](/img/attention-is-all-you-need/01.png)

“어떤 정보를 찾아야 하는가”(query)와 encoder에서 입력 받은 문장의 모든 단어(key,답)를 비교하여 단어 간 관련성에 따라 attention score 출력. 여기에 단어들의 정보(Value)를 곱하여 이를 반영한 정보가 새로운 벡터를 생성
→이 높은 attention score에 정보가 곱해진 벡터를 통해 다양한 task 진행
**Ex)**
1. 긍정 부정 분류기
"I really love this movie. It was amazing!"
→**"love", "amazing" 같은 단어**에 높은 가중치를 주고 "긍정적"으로 분류함.
1. 질의 응답
질문: "Where was Albert Einstein born?"
지문: "... Albert Einstein was born in Ulm, Germany in 1879 ..."(사전 학습됨)
질문(query)와 지문(key)을 비교하며 attention score구하고 이를 value에 곱해서 각 단어들의 관련성에 의해 나온 점수가 높은 단어를 출력하는 방식
**핵심: 질문과 답 간의 관련성들을 구한 벡터로 결과를 출력해낸다. 단순히 a→b 매칭보다는 각 단어들 끼리의 관계성도 영향을 준다.**

## **Attention is all you  need 과정**

1. 문장 입력 시 문장을 토큰화(the cat is playing on ther mat→ the, cat, is, playing) 토크나이저로 토큰화
1. vocab을 사용하여 토큰에 대응하는 ID로 변환
1. 임베딩 레이어에 넣어서 고차원 벡터로 변경

#### **Positional Encoding 추가**

**사용 이유:**
Transformer은 단어를 한번에 병렬적 처리
→단어 간 순서 정보 손실 문제 발생
**적용 방식:**
토큰에 위치 정보 값 더하는 방식

![그림 2](/img/attention-is-all-you-need/02.png)

![그림 3](/img/attention-is-all-you-need/03.png)

벡터의 차원 수가 매우 크기에 식을 사용한 값을 더하게 될 시 각 위치가 특정될 수 있다.

## self_attention

자기 자신을 q,k,v에 넣음
(입력 문장의 모든 단어를 서로 비교해서 각 단어가 다른 단어와 얼마나 관련이 있는지 학습하는 과정)

![그림 4](/img/attention-is-all-you-need/04.png)

q, k를 내적하여 각 단어 별 단어에 대한 중요도에 따른 확률로 변환, 이걸

![그림 5](/img/attention-is-all-you-need/05.png)

해서 중요도 높은 값의 v(단어)이 많이 반영된 context 벡터 생성(출력될 단어가 입력된 단어 각각에 대해 어느 정도 초점을 맞춰야 하는 지에 대한 정보 반영)
**→이 vector가 다음 self-attention에서 사용됨.**
**여러 개의 Self-Attention 레이어를 거치면서, 학습이 진행됨**

잔차
위 벡터를 원본 입력 벡터와 더하고 normalization함.
20250319 수정 필요 FFN 거침
단어들을 개별 벡터로 변형하고

![그림 6](/img/attention-is-all-you-need/06.png)

를 거쳐서 ~~~~
이후 또 skipconnection으로

![그림 7](/img/attention-is-all-you-need/07.png)

거쳐서 h’ 정보 추가함.

## Decoder

이 단계에선 Masked_self_attention과 Cross-Attention이 진행됨. (순서대로 표기해놓음)
ex) [start] i am a student [EOS] ([start] 기준 예시)
1. Self-Attention(Masked)
  1. 정답 값 q, k, v(현재 상태: q :[start], v: [start], k: [start], 나머지 mask)
  1. 디코더가 예측한 단어들만 masked가 풀림(현재 상태: start만 unmasked, 이후엔 풀림)
    1. 특이점: 틀려도 정답 값이 masked됨
    1. start로 시작점, EOS로 멈출 시점 정의
  1. unmasked 된 단어들을 통해 self-attention해서 관계 학습한 vector 생성
  1. vector를 cross-attention에 보냄
→ 디코더가 생성한 단어들 context 학습
1. Cross-Attention (인코더 정보 활용)
  1. 1. d 에서 받은 vector를 query에 넣음. (현재 상태: q :[start], v: , k: encoder의 context vector)
    1. 특이점: q는 mask된 self의 학습 q값을 받음
  1. attention을 통해 vector 생성(인코더의 원본 정보 반영)
→ decoder의 context+ 원본의 정보 반영된 vector 생성

1. FFN
  1. 2.b의 vector를 FFN에 넣음

![그림 8](/img/attention-is-all-you-need/08.png)

    1. 차원 확장, ReLU로 필요 없는 정보 날림, 차원 축소 구조로 중요 정보 강조
1. 단어 예측
  1. FFN 거친 벡터를 아래 선형 함수에 넣어 단어에 맞는 점수(확률 분포)로 변환

![그림 9](/img/attention-is-all-you-need/09.png)

    1. W(vocab)는 모든 단어에 대한 매핑 역할을 함.
  1. 위 logits 값을 softmax에 넣어 단어에 대한 확률로 바꿈

![그림 10](/img/attention-is-all-you-need/10.png)

1. 가장 확률 높은 단어를 다음 단어로 사용
1. self-attention 과정에 다음 단어를 추가하여 위 과정 반복

![그림 11](/img/attention-is-all-you-need/11.png)

## Multi-Head Attention

단순히 embedding(고차원 벡터변환)할 때 단순히 임베딩 사이즈를 헤드 수로 나눠서 사용하는 것 (Multi-head 수 만큼)
→이걸 d_k *num_heads 를 projection 하고 view()로 head 단위 분리

![그림 12](/img/attention-is-all-you-need/12.png)

```text
   x.shape = [batch_size, seq_len, d_model]
   self.q_linear = nn.Linear(d_model,d_k*num_head)

def forward(self, x):
    q = self.q_linear(x).view(x.size(0),x.size(1),self.h,self.d_k).transpose(1,2)
```

- 입력 x.shape == [batch_size, seq_len, d_model]임
- linear()통과⇒ [batch_size, seq_len, d_k*num_head]
  - linear(a,b)는 마지막 차원이 a인 벡터 받아서 b로 변환
- batch_size*len 은 직사각형을 여러개 붙여놓은 형태, d_k*num_head를 곱해서 직육면체로 변환.
  - 즉, 기존 x를 q,k의 차원에 맞게 변환 후 헤드 수만큼 크기를 키움(차원 키움, head 수만큼)
- view를 통해 [batch, seq_len, num_heads * head_dim]을 [batch_size, seq_len, num_heads, head_dim]로 변경
  - 키운 차원을 헤드수로 나눠서 forward함.
- transpose(1, 2)→1, 2 번 위치 변환
출력 벡터 차원마다 하나의 편향이 필요
→ num_heads *head_dim 으로 각 출력 값마다 하나씩 편향을 더함.

d_model= 임베딩 차원 수.

![그림 13](/img/attention-is-all-you-need/13.png)

![그림 14](/img/attention-is-all-you-need/14.png)

![그림 15](/img/attention-is-all-you-need/15.png)
