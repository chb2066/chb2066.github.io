<!--
개정: 2026-09-10 (원본: src/content/reviews/attention-is-all-you-need.md)
- ⚠️ 원본에 미완성 표시 세 곳이 있었다. 논문 3.1~3.3절로 채웠다.
  · "20250319 수정 필요 FFN 거침"
  · "를 거쳐서 ~~~~"
  · "거쳐서 h' 정보 추가함"
  → encoder 블록의 두 sub-layer 구조(self-attention → FFN)와
    각각을 감싸는 residual + LayerNorm 을 명시적으로 서술했다.
    FFN 은 d_model=512 → d_ff=2048 → 512 의 두 선형 변환에 ReLU 를 끼운 것이고,
    각 sub-layer 는 LayerNorm(x + Sublayer(x)) 형태다
- 「전체 구조」 신설 — encoder/decoder 각 6층, d_model=512. 원본에 층 수가 없어서
  "여러 개의 Self-Attention 레이어" 로만 서술돼 있었다
- 「실험에서 확인된 것」 신설 — WMT2014 En-De 28.4 BLEU, En-Fr 41.8 BLEU,
  8 GPU 3.5일. 원본에 결과가 아예 없었다
- 오타 수정: "ther mat" → "the mat"
- 끝맺음을 평서형으로 통일
- 원본의 attention 직관 설명(긍정·부정 분류, 질의응답 예시)과 decoder 단계별 추적,
  multi-head 의 view/transpose 코드 설명은 그대로 유지. 이 글에서 가장 좋은 부분이다
-->
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

핵심 키워드:
self-attention, multi-head, positional encoding, encoder-decoder
주요 전략:
순환 구조를 없애고 attention만으로 시퀀스를 처리해, 병렬화와 장거리 의존성을 동시에 얻는다
사용 가능 분야:
기계 번역을 시작으로 사실상 모든 시퀀스 모델링

#### 사전 개념 — attention

![그림 1](/img/attention-is-all-you-need/01.png)

**"어떤 정보를 찾아야 하는가"(query)**와 encoder에서 입력받은 문장의 모든 단어(key)를 비교해 단어 간 관련성에 따라 attention score를 출력한다. 여기에 단어들의 정보(value)를 곱해서 이를 반영한 새로운 벡터를 만든다.

이렇게 만들어진, **높은 attention score에 정보가 곱해진 벡터**로 다양한 태스크를 수행한다.

**예시 1 — 긍정/부정 분류기**

```text
"I really love this movie. It was amazing!"
```

**"love", "amazing" 같은 단어**에 높은 가중치를 주고 "긍정적"으로 분류한다.

**예시 2 — 질의응답**

```text
질문: "Where was Albert Einstein born?"
지문: "... Albert Einstein was born in Ulm, Germany in 1879 ..."
```

질문(query)과 지문(key)을 비교해 attention score를 구하고, 이를 value에 곱해서 관련성이 높은 단어를 출력한다.

> **핵심은 질문과 답 사이의 관련성을 구한 벡터로 결과를 출력한다는 것이다.**
> 단순한 a→b 매칭이 아니라 각 단어들끼리의 관계성이 함께 영향을 준다.

#### 입력 처리

1. 문장을 토큰화한다. `the cat is playing on the mat` → `the`, `cat`, `is`, `playing`, …
2. vocab을 사용해 토큰에 대응하는 ID로 변환한다.
3. 임베딩 레이어에 넣어 고차원 벡터로 바꾼다.

**Positional Encoding**

*왜 필요한가* — Transformer는 단어를 한 번에 병렬 처리하므로 **단어 간 순서 정보가 사라진다.**

*방식* — 토큰에 위치 정보 값을 더한다.

![그림 2](/img/attention-is-all-you-need/02.png)

![그림 3](/img/attention-is-all-you-need/03.png)

벡터의 차원 수가 매우 크기 때문에, 이 식으로 만든 값을 더하면 각 위치가 특정될 수 있다.

#### Self-attention

자기 자신을 query, key, value에 모두 넣는다. 입력 문장의 모든 단어를 서로 비교해서 각 단어가 다른 단어와 얼마나 관련이 있는지 학습하는 과정이다.

![그림 4](/img/attention-is-all-you-need/04.png)

query와 key를 내적해 각 단어별 중요도를 확률로 변환한다.

![그림 5](/img/attention-is-all-you-need/05.png)

이 확률로 가중합하면 **중요도가 높은 value가 많이 반영된 context 벡터**가 만들어진다. 출력될 단어가 입력된 각 단어에 어느 정도 초점을 맞춰야 하는지가 반영된다.

이 벡터가 다음 self-attention 층에서 다시 쓰인다.

#### Encoder 블록의 구조

encoder는 **동일한 층 6개를 쌓은 구조**다. 그리고 각 층은 **두 개의 sub-layer**로 이루어진다.

**sub-layer 1 — Multi-Head Self-Attention**
위에서 설명한 self-attention이다.

**sub-layer 2 — Position-wise Feed-Forward Network**

attention sub-layer에 더해, 각 층은 완전연결 feed-forward 네트워크를 갖는다. 이 네트워크는 **각 위치에 대해 개별적으로, 그리고 동일하게** 적용된다. 단어들을 각각 독립적인 벡터로 변형하는 것이다.

구성은 **ReLU를 사이에 끼운 두 개의 선형 변환**이다.

```text
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
```

![그림 6](/img/attention-is-all-you-need/06.png)

차원을 보면 의도가 분명하다.

- 입력과 출력의 차원: `d_model = 512`
- 내부 층의 차원: `d_ff = 2048`

**차원을 4배로 확장했다가 ReLU로 불필요한 정보를 날리고 다시 축소**하는 구조다. 중요한 정보를 강조하는 효과가 있다.

**두 sub-layer를 감싸는 것 — residual + LayerNorm**

각 sub-layer의 출력은 그대로 다음으로 가지 않는다. **입력을 더하고 정규화한다.**

```text
출력 = LayerNorm(x + Sublayer(x))
```

즉 흐름은 이렇다.

```text
x  →  Multi-Head Self-Attention  →  + x  →  LayerNorm  →  h
h  →  Feed-Forward Network       →  + h  →  LayerNorm  →  h'
```

![그림 7](/img/attention-is-all-you-need/07.png)

skip connection으로 원본 신호를 우회시켜 더하는 것이고, 이렇게 해서 `h'`에 정보가 누적된다.

residual 연결을 쓰기 위해 **모델의 모든 sub-layer와 임베딩 층이 동일한 차원 `d_model = 512`로 출력**한다. 차원이 같아야 더할 수 있기 때문이다.

#### Decoder

Masked Self-Attention과 Cross-Attention이 순서대로 진행된다. decoder도 6층이고, encoder의 두 sub-layer에 **encoder 출력에 대한 attention을 수행하는 세 번째 sub-layer**가 추가된다.

예시: `[start] i am a student [EOS]` — `[start]` 기준으로 따라가 본다.

**1. Masked Self-Attention**
- 정답값을 query, key, value에 넣는다. 현재 상태는 `q: [start]`, `k: [start]`, `v: [start]`이고 나머지는 mask된다.
- **디코더가 예측한 단어들만 mask가 풀린다.** 지금은 `start`만 unmasked이고 이후 순차적으로 풀린다.
  - 특이점: 틀려도 정답값이 mask된다.
  - `start`로 시작점을, `EOS`로 멈출 시점을 정의한다.
- unmasked된 단어들로 self-attention해서 관계를 학습한 벡터를 만든다.
- 이 벡터를 cross-attention으로 보낸다.

→ **디코더가 지금까지 생성한 단어들의 context를 학습한다.**

**2. Cross-Attention**
- 앞 단계에서 받은 벡터를 query에 넣는다. 현재 상태는 `q: [start]`, `k`와 `v`: encoder의 context vector다.
  - 특이점: query는 mask된 self-attention에서 학습된 값을 받는다.
- attention으로 벡터를 만든다. encoder의 원본 정보가 반영된다.

→ **디코더의 context와 원본 정보가 함께 반영된 벡터가 나온다.**

**3. FFN**
앞 단계의 벡터를 FFN에 넣는다.

![그림 8](/img/attention-is-all-you-need/08.png)

차원 확장 → ReLU로 필요 없는 정보 제거 → 차원 축소 구조로 중요 정보를 강조한다.

**4. 단어 예측**
FFN을 거친 벡터를 선형 함수에 넣어 단어에 맞는 점수(logits)로 변환한다.

![그림 9](/img/attention-is-all-you-need/09.png)

`W_vocab`은 모든 단어에 대한 매핑 역할을 한다. 이 logits를 softmax에 넣어 단어에 대한 확률로 바꾼다.

![그림 10](/img/attention-is-all-you-need/10.png)

**5. 반복**
가장 확률이 높은 단어를 다음 단어로 쓰고, self-attention 과정에 그 단어를 추가해 위 과정을 반복한다.

![그림 11](/img/attention-is-all-you-need/11.png)

#### Multi-Head Attention

임베딩 크기를 head 수로 나눠서 쓰는 것이다. `d_k × num_heads`로 projection한 뒤 `view()`로 head 단위로 분리한다.

![그림 12](/img/attention-is-all-you-need/12.png)

```python
x.shape = [batch_size, seq_len, d_model]
self.q_linear = nn.Linear(d_model, d_k * num_head)

def forward(self, x):
    q = self.q_linear(x).view(x.size(0), x.size(1), self.h, self.d_k).transpose(1, 2)
```

- 입력 `x.shape == [batch_size, seq_len, d_model]`이다.
- `linear()`를 통과하면 `[batch_size, seq_len, d_k * num_head]`가 된다. `Linear(a, b)`는 마지막 차원이 `a`인 벡터를 받아 `b`로 변환한다.
- `batch_size × seq_len`은 직사각형을 여러 개 붙여놓은 형태이고, 여기에 `d_k * num_head`를 곱해 직육면체로 만든다. 즉 기존 `x`를 query, key의 차원에 맞게 변환한 뒤 head 수만큼 차원을 키우는 것이다.
- `view`로 `[batch, seq_len, num_heads * head_dim]`을 `[batch, seq_len, num_heads, head_dim]`으로 바꾼다. 키운 차원을 head 수로 나눠 forward한다.
- `transpose(1, 2)`로 1번과 2번 위치를 교환한다.

출력 벡터의 차원마다 하나의 편향이 필요하므로, `num_heads × head_dim`으로 각 출력값마다 하나씩 편향을 더한다.

`d_model`은 임베딩 차원 수다.

![그림 13](/img/attention-is-all-you-need/13.png)

![그림 14](/img/attention-is-all-you-need/14.png)

![그림 15](/img/attention-is-all-you-need/15.png)

#### 실험에서 확인된 것

WMT 2014 기계 번역 태스크 기준이다.

| 태스크 | BLEU |
|---|---|
| English → German | **28.4** — 기존 최고 성능(앙상블 포함)을 2 BLEU 이상 상회 |
| English → French | **41.8** — 단일 모델 최고 성능 |

English→French 모델은 **GPU 8장으로 3.5일** 학습했다. 당시 최고 모델들의 학습 비용에 비하면 아주 작은 부분이다.

**속도가 이 논문의 핵심 주장 중 하나**다. 순환 구조가 없으므로 시퀀스 전체를 병렬로 처리할 수 있고, 그래서 같은 성능을 훨씬 적은 학습 시간에 도달한다.

#### 정리

이 논문이 없앤 것은 **순환**이다. 그리고 순환이 하던 두 가지 일을 각각 다른 것으로 대체했다.

- **순서 정보** → positional encoding
- **장거리 의존성** → self-attention

순환을 없애면 병렬화가 가능해지고, attention은 거리에 무관하게 연결하므로 오히려 장거리 의존성을 더 잘 잡는다. 잃은 것 없이 얻기만 한 것처럼 보이는데, 대가는 **길이의 제곱에 비례하는 비용**이다. 이후 연구들의 상당 부분이 그 비용을 줄이는 데 쓰인다.
