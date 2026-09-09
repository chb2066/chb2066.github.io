<!--
개정: 2026-09-10 (원본: src/content/reviews/bert.md)
- 오타·미완성 수정
  · "이용하여 이용하여" 중복 제거
  · "10%는 그대로 유" → "10%는 그대로 유지한다"
  · "독립 테스크 동시 작동하고 각자 계산 뒤 합" → 문장으로 완성
- 「왜 80/10/10 인가」 보강 — 원본은 비율만 적었는데, 이 배분이 있는 이유
  (파인튜닝 시점에는 [MASK] 토큰이 없으므로 사전학습과 파인튜닝 사이의 불일치를 줄여야 한다)
  가 빠져 있었다. 논문 3.1절
- 「배경 지식」에 양방향이 왜 어려운지 추가 — 표준 언어모델은 단방향이어야 하고,
  양방향으로 두면 토큰이 자기 자신을 간접적으로 볼 수 있다. MLM 이 그 우회책이다
- 「실험에서 확인된 것」 신설 — GLUE 80.5%, SQuAD v1.1 F1 93.2 등. 원본에 결과가 없었다
- i-jepa 형식으로 재배치
- 끝맺음을 평서형으로 통일
- 원본의 임베딩 3종 코드 설명은 그대로 유지. 구현 관점의 설명이 좋다
-->
---
title: BERT
paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
venue: NAACL 2019
link: https://arxiv.org/abs/1810.04805
claim: MLM과 NSP로 양방향 문맥을 사전학습하면 적은 자원으로 다양한 downstream task에 전이된다.
tags: [NLP, Pretraining]
tier: basic
date: 2025-03-28
draft: true
---

핵심 키워드:
양방향 사전학습, Masked Language Model, Next Sentence Prediction, 전이 학습
주요 전략:
일부 토큰을 가리고 맞히게 해서, 단방향 제약 없이 양쪽 문맥을 모두 쓰는 표현을 학습한다
사용 가능 분야:
분류, 질의응답, 개체명 인식 등 대부분의 자연어 이해 태스크

#### 배경 지식

**전이 학습**
사전 학습된 모델이 대용량의 라벨링되지 않은 데이터로 언어 모델을 학습하고, 이를 토대로 특정 작업을 위한 신경망을 추가하는 방식이다.

**사전 학습 모델**
BERT는 기본적으로 대량의 단어 임베딩에 대해 사전 학습된 모델을 제공한다. 따라서 상대적으로 적은 자원만으로도 자연어 처리의 여러 작업을 수행할 수 있다.

**왜 양방향이 어려운가**

표준 언어 모델은 **단방향**일 수밖에 없다. 다음 단어를 예측하는 과제이므로 미래를 보면 안 되기 때문이다. 그런데 양쪽 문맥을 모두 보는 편이 이해 태스크에는 유리하다.

문제는 **양방향 Transformer를 그냥 쌓으면 각 토큰이 여러 층을 거치며 간접적으로 자기 자신을 볼 수 있다**는 점이다. 그러면 예측이 무의미해진다.

BERT는 이를 **Masked Language Model**로 우회한다. 일부 토큰을 가리고 그것만 맞히게 하면, 나머지 전체 문맥을 양방향으로 써도 된다.

#### 구조

세 가지 임베딩의 합으로 입력이 구성된다. 셋 모두 **학습 가능한** 임베딩이다. `nn.Embedding(num_embeddings, embedding_dim)`을 쓰고 내부에 파라미터가 있다.

![그림 1](/img/bert/01.png)

```python
import torch
import torch.nn as nn

class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeddings = self.token_embeddings(input_ids)
        pos_embeddings = self.position_embeddings(position_ids)
        seg_embeddings = self.segment_embeddings(token_type_ids)

        embeddings = word_embeddings + pos_embeddings + seg_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
```

**Token Embeddings**
텍스트의 토큰 ID를 임베딩 벡터로 변환한다.

```python
self.token_embeddings = nn.Embedding(vocab_size, hidden_size)

# vocab_size  = tokenizer 에 존재하는 전체 어휘 사전의 크기
#               예: [CLS], apple, ##ing, [SEP]
# hidden_size = 각 토큰을 임베딩할 때 사용하는 차원 수.
#               BERT 구조에 따라 고정된다 (bert-base-uncased, bert-large-uncased 등)
```

![그림 2](/img/bert/02.png)

**Segment Embedding**

```python
self.segment_embeddings = nn.Embedding(type_vocab_size, hidden_size)
# 0, 1 로 구분된다
```

사전학습 시 `[SEP]` 토큰으로 문장을 분리하고, 문장 구분을 위해 0과 1로 표시한다. 이를 통해 다음 문장이 맞는 문장인지 예측하는 NSP를 수행한다.

**두 개의 문장까지만 사용할 수 있으므로 두 문장씩 잘라서 학습한다.**

**Position Embedding**

```python
self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
# max_position_embeddings 는 한 번에 처리 가능한 최대 토큰 수
```

각 위치에 해당하는 고유한 위치 벡터다.

![그림 3](/img/bert/03.png)

#### 사전학습

**NSP (Next Sentence Prediction)**

두 문장씩 끊어서 학습한다. 두 문장의 관계를 이해하기 위해, **두 번째 문장이 첫 문장의 바로 다음에 오는 문장인지** 예측한다.

- 50%는 실제로 이어지는 문장을 쓴다.
- 50%는 전체 말뭉치에서 나온 임의의 문장을 쓴다.

`[CLS]` 토큰의 출력을 2차원 벡터로 변환하고, `IsNext` 라벨을 softmax로 할당한다.

**MLM (Masked Language Model)**

전체 토큰의 **15% 정도를 마스킹**한다. 그리고 그 15% 안에서 다시 나눈다.

| 비율 | 처리 |
|---|---|
| 80% | 토큰을 `[MASK]`로 바꾼다 |
| 10% | 토큰을 무작위 단어로 바꾼다 |
| 10% | **그대로 유지한다** |

마스킹된 위치만 예측하게 하므로 문맥에 대한 이해도가 올라간다. 마스크 토큰 복원은 softmax로 한다.

**왜 80/10/10으로 나누는가**

`[MASK]` 토큰은 **사전학습에만 등장하고 파인튜닝 시점에는 나타나지 않는다.** 100%를 `[MASK]`로 바꾸면 모델이 "마스크 자리만 신경 쓰면 된다"고 학습해버려서, 실제 문장을 다룰 때 표현이 약해진다.

- **무작위 단어 10%** — 모델이 어떤 토큰이든 문맥으로 검증하게 만든다.
- **원본 유지 10%** — 마스크 표시가 없어도 그 위치의 표현을 제대로 만들도록 강제한다.

사전학습과 파인튜닝 사이의 불일치를 줄이는 장치다.

**전체 손실**

MLM과 NSP는 **독립적인 태스크로 동시에 수행되며, 각각 계산한 뒤 합산한다.**

```text
Loss_total = Loss_MLM + Loss_NSP
```

#### Fine-tuning

최대 토큰 수만큼 끊어서 학습한다. 사전학습된 파라미터 위에 태스크별 출력 층 하나만 얹으면 되므로, **구조를 크게 바꾸지 않고 다양한 태스크로 전이**된다.

#### 실험에서 확인된 것

| 벤치마크 | 성능 |
|---|---|
| GLUE | **80.5%** (기존 대비 +7.7%p) |
| SQuAD v1.1 | F1 **93.2** |
| SQuAD v2.0 | F1 **83.1** |
| MultiNLI | **86.7%** |

11개 자연어 처리 태스크에서 당시 최고 성능을 기록했다. **구조를 태스크마다 새로 설계하지 않고 같은 사전학습 모델에 출력 층만 바꿔 얹었다**는 점이 함께 강조된다.

#### 정리

BERT가 한 일은 **"양방향으로 보면 안 된다"는 제약을 과제 설계로 우회한 것**이다.

다음 단어를 맞히는 과제는 본질적으로 단방향을 요구한다. 그런데 **일부를 가리고 그것만 맞히는 과제**로 바꾸면 나머지 전체를 양방향으로 볼 수 있다. 제약은 과제에서 나온 것이지 모델에서 나온 게 아니었다는 뜻이다.

그리고 80/10/10 배분은 **사전학습과 배포 사이의 분포 불일치**를 다루는 사례다. 학습 시점에만 존재하는 인공물(`[MASK]`)에 모델이 의존하지 않게 만드는 것이고, 같은 문제는 다른 곳에서도 반복된다.
