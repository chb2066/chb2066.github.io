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

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Abstract

**문제**

기존 시퀀스 변환 모델은 RNN/CNN 인코더-디코더에 attention을 얹는 구조. 순환 구조가 순차 계산을 강제해 학습을 병렬화할 수 없고, 시퀀스가 길어질수록 메모리 제약이 심해짐

**해결책**

- 순환과 합성곱을 전부 제거하고 attention만으로 구성한 Transformer 제안
- 순서 정보는 positional encoding으로 별도 주입, 서로 다른 관계는 multi-head로 동시에 포착

---

## 1. Introduction

**attention이란**: "어떤 정보를 찾아야 하는가"(query)와 문장의 모든 단어(key)를 비교해 관련성 점수를 내고, 그 점수로 단어들의 정보(value)를 가중합해 새 벡터를 만드는 연산

![그림](/img/attention-is-all-you-need/01.png)

**쓰이는 곳**
- **긍정/부정 분류** - `"I really love this movie. It was amazing!"` 에서 `love`, `amazing` 에 높은 가중치를 주고 긍정으로 분류
- **질의응답** - 질문(query)과 지문(key)을 비교해 관련성이 높은 단어를 출력

> 핵심은 질문과 답 사이의 관련성을 구한 벡터로 결과를 출력한다는 것이다.
> 단순한 a→b 매칭이 아니라 각 단어들끼리의 관계성이 함께 영향을 준다.

**기존 상황**
- RNN 계열은 이전 상태를 받아야 다음을 계산 → 순차 계산이 강제됨
- attention은 이미 쓰이고 있었지만 대부분 순환 구조에 덧붙이는 형태
- 순환이 남아 있는 한 병렬화도 장거리 의존성도 한계

**본 논문의 기여**
1. 순환·합성곱을 전부 빼고 attention만으로 구성
1. 학습을 완전히 병렬화 → 같은 성능을 훨씬 적은 학습 시간에
1. WMT 2014 영→독 28.4 BLEU 로 기존 최고 성능(앙상블 포함)을 2 BLEU 이상 상회

---

## 2. Background

**순차 계산을 줄이려는 기존 시도**
- Extended Neural GPU, ByteNet, ConvS2S - 합성곱을 기본 블록으로 써서 병렬화 시도
- 한계: 두 위치의 신호를 잇는 데 필요한 연산이 거리에 따라 늘어남(선형 또는 로그) → 먼 의존성을 배우기 어려움

**Transformer의 차이**
- 두 위치를 잇는 연산이 거리와 무관하게 상수 → 장거리 의존성에 유리
- 대가: attention 가중 평균이 해상도를 떨어뜨림 → multi-head로 상쇄

---

## 3. Model Architecture

**입력 요소**
1. 문장을 토큰화 - `the cat is playing on the mat` → `the`, `cat`, `is`, `playing`, …
1. vocab으로 토큰을 ID로 변환
1. 임베딩 레이어를 거쳐 고차원 벡터로

### 3.1 Encoder and Decoder Stacks

**Encoder** - 동일한 층 6개를 쌓은 구조. 각 층은 두 개의 sub-layer로 이뤄진다.

- **sub-layer 1** - Multi-Head Self-Attention
- **sub-layer 2** - Position-wise Feed-Forward Network

**두 sub-layer를 감싸는 것 - residual + LayerNorm**
```text
출력 = LayerNorm(x + Sublayer(x))

x  →  Multi-Head Self-Attention  →  + x  →  LayerNorm  →  h
h  →  Feed-Forward Network       →  + h  →  LayerNorm  →  h'
```
- skip connection으로 원본 신호를 우회시켜 더함 → `h'` 에 정보가 누적됨
- residual 을 쓰려면 차원이 같아야 함 → 모든 sub-layer와 임베딩 층이 `d_model = 512` 로 출력

![그림](/img/attention-is-all-you-need/07.png)

**Decoder** - 6층이고, encoder 의 두 sub-layer 에 encoder 출력에 대한 attention 을 수행하는 세 번째 sub-layer 가 추가된다.

예시 `[start] i am a student [EOS]` 를 `[start]` 기준으로 따라가면,

**Masked Self-Attention**
- 정답값을 query, key, value 에 넣음. 현재 `q: [start]`, `k: [start]`, `v: [start]` 이고 나머지는 mask 됨
- 디코더가 예측한 단어들만 mask 가 풀림 → 지금은 `start` 만 unmasked, 이후 순차적으로 풀림
- unmasked 된 단어들로 self-attention 해서 관계를 학습한 벡터를 만듦
→ 디코더가 지금까지 생성한 단어들의 context 를 학습

**Cross-Attention**
- 앞 단계 벡터를 query 에, encoder 의 context vector 를 key·value 에 넣음
- query 는 mask 된 self-attention 에서 학습된 값을 받음
→ 디코더의 context 와 원본 정보가 함께 반영된 벡터가 나옴

**FFN → 단어 예측 → 반복**
- FFN 을 거친 벡터를 선형 함수에 넣어 단어 점수(logits)로 변환. `W_vocab` 이 모든 단어에 대한 매핑
- softmax 로 확률로 바꾸고, 가장 확률이 높은 단어를 다음 단어로 씀
- 그 단어를 self-attention 과정에 추가해 반복

![그림](/img/attention-is-all-you-need/11.png)

### 3.2 Attention

**Self-attention** - 자기 자신을 query, key, value 에 모두 넣는다. 입력 문장의 모든 단어를 서로 비교해 각 단어가 다른 단어와 얼마나 관련 있는지 학습한다.

- query 와 key 를 내적해 각 단어별 중요도를 확률로 변환
- 그 확률로 value 를 가중합 → 중요도가 높은 value 가 많이 반영된 context 벡터
- 출력될 단어가 입력된 각 단어에 어느 정도 초점을 맞춰야 하는지가 반영됨
- 이 벡터가 다음 self-attention 층에서 다시 쓰임

![그림](/img/attention-is-all-you-need/05.png)

**Multi-Head Attention** - 임베딩 크기를 head 수로 나눠서 쓴다. `d_k × num_heads` 로 projection 한 뒤 head 단위로 분리한다.

```python
x.shape = [batch_size, seq_len, d_model]
self.q_linear = nn.Linear(d_model, d_k * num_head)

def forward(self, x):
    q = self.q_linear(x).view(x.size(0), x.size(1), self.h, self.d_k).transpose(1, 2)
```

- `linear()` 를 통과하면 `[batch, seq_len, d_k * num_head]` → 기존 `x` 를 query·key 차원에 맞게 변환한 뒤 head 수만큼 차원을 키움
- `view` 로 `[batch, seq_len, num_heads, head_dim]` 으로 쪼갬 → 키운 차원을 head 수로 나눠 forward
- `transpose(1, 2)` 로 축 교환
- 출력 벡터의 차원마다 편향이 하나씩 필요 → `num_heads × head_dim` 만큼 더함

![그림](/img/attention-is-all-you-need/12.png)

### 3.3 Position-wise Feed-Forward Networks

attention sub-layer 에 더해 각 층은 완전연결 FFN 을 갖는다. 각 위치에 개별적으로, 그리고 동일하게 적용된다 - 단어들을 각각 독립적인 벡터로 변형하는 것.

```text
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
```

- 입력·출력 차원 `d_model = 512`, 내부 층 차원 `d_ff = 2048`
- 차원을 4배로 확장했다가 ReLU 로 불필요한 정보를 날리고 다시 축소 → 중요한 정보를 강조

### 3.5 Positional Encoding

- **필요한 이유** - Transformer 는 단어를 한 번에 병렬 처리하므로 단어 간 순서 정보가 사라짐
- **방식** - 토큰 임베딩에 위치 정보 값을 더함
- 벡터 차원 수가 매우 크기 때문에, 이 식으로 만든 값을 더하면 각 위치가 특정될 수 있음

![그림](/img/attention-is-all-you-need/02.png)

![그림](/img/attention-is-all-you-need/03.png)

---

## 4. Why Self-Attention

순환·합성곱과 비교해 세 가지를 따진다.

- **층당 계산 복잡도**
- **병렬화 가능한 연산량** - 순차적으로 처리해야 하는 최소 연산 수
- **장거리 의존성의 경로 길이** - 두 위치를 잇는 데 필요한 연산 수. 짧을수록 먼 의존성을 배우기 쉬움

self-attention 은 경로 길이가 상수라 세 번째에서 압도적이다. 대신 시퀀스 길이의 제곱에 비례하는 비용을 치른다.

---

## 6. Results

WMT 2014 기계 번역 기준이다.

| 태스크 | BLEU |
|---|---|
| English → German | 28.4 - 기존 최고 성능(앙상블 포함)을 2 BLEU 이상 상회 |
| English → French | 41.8 - 단일 모델 최고 성능 |

- English→French 모델은 GPU 8장으로 3.5일 학습. 당시 최고 모델들의 학습 비용에 비하면 아주 작은 부분
- **속도가 이 논문의 핵심 주장 중 하나** - 순환이 없으므로 시퀀스 전체를 병렬 처리할 수 있고, 그래서 같은 성능에 훨씬 적은 학습 시간

**attention 시각화**

- encoder self-attention 에서 장거리 의존성을 따라가는 head 가 실제로 관찰됨
