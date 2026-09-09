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

#### **전이 학습 **

사전 학습된 모델이 대용량의 레이블링 되지 않는 데이터를 이용하여
이용하여 언어 모델을 학습하고 이를 토대로 특정 작업을 위한
신경망을 추가하는 방식

#### **사전 학습 모델**

BERT는 기본적으로 대량의 단어 임베딩에 대해 사전 학습이 된 모델을 제공함 따라서 상대적으로 적은 자원만으로도 충분히 자연어 처리의 여러 일 수행 가능

#### **구조**

![그림 1](/img/bert/01.png)

세 임베딩 합으로 구성. 세 임베딩 모두 학습 가능한  embedding 을 사용한다.(nn.Embedding 사용, nn.Embedding(num_embeddings, embedding_dim)을 의미하고 내부에 파라미터가 존재)

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
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (batch_size, seq_length)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 각 임베딩 가져오기
        word_embeddings = self.token_embeddings(input_ids)
        pos_embeddings = self.position_embeddings(position_ids)
        seg_embeddings = self.segment_embeddings(token_type_ids)

        embeddings = word_embeddings + pos_embeddings + seg_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

```

**Token Embeddings**
텍스트의 토큰 id를 임베딩 벡터로 변환

```python
self.token_embeddings = nn.Embedding(vocab_size, hidden_size)

#vocab_size= tokenizer에 존재하는 전체 어휘 사전의 크기
#ex) [cls], apple, ##ing, [sep]
#hidden_size=각 토큰들을 임베딩할 떄 사용되는 차원 수,
#BERT 구조에 따라 고정
#(bert-base-uncased, bert-large-uncased 등등)
```

![그림 2](/img/bert/02.png)

Segment Embedding

```python
self.segment_embeddings = nn.Embedding(type_vocab_size, hidden_size)
##0, 1로 구분됨
```

pretrain 시 sep 토큰으로 문장 분리, 문장 구분을 위해 0, 1로 문장을 구분한다. 이를 통해 다음 문장이 맞는 문장인지를 예측하는 NSP를 수행함.
두 개의 문장까지만 사용이 가능하기 때문에 두 문장 씩 잘라서 학습함
Position Embedding

```python
self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
#max_position_embedding은 한 번에 처리 가능 최대 토큰 수
```

각 위치에 해당하는 고유한 위치 벡터

![그림 3](/img/bert/03.png)

**Pre-training**
NSP
2문장씩 끊어서 학습
두 문장의 관계 이해를 위해 두 번째 문장이 첫 문장의 바로 다음에 오는 문장인지 예측.
50%는 실제 문장, 50%는 전체 말뭉치에서 나오는 임의의 문장
이를 통해 두 번째 문장이 임의의 문장인지 여부를 예측한다.
CLS 토큰의 출력은 2*1 벡터로 변환
IsNext_label은 softmax로 할당

MLM
15%정도 Masking함(이 중 80%는 토큰을 mask로 10%는 토큰을 무작위 단어로 바꿈. 10%는 그대로 유).
mask만을 예측함→문맥에 대한 이해도를 높이기 위함.
Mask 토큰 복원 softmax
독립 테스크 동시 작동하고 각자 계산 뒤 합
전체 loss=Loss_MLM+Loss_NSP
**Fine Tuning **
max_token수만큼 끊어서 학습
