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

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## Abstract

**문제**
표준 언어모델은 다음 단어를 맞히는 과제라 단방향일 수밖에 없음. 그런데 이해 태스크에는 양쪽 문맥을 다 보는 편이 유리함

**해결책**
일부 토큰을 가리고 그것만 맞히는 Masked Language Model로 과제를 바꿔, 나머지 전체 문맥을 양방향으로 쓸 수 있게 함
사전학습 모델 위에 출력 층 하나만 얹어 11개 태스크에 전이

---

## 1. Introduction

**전이 학습이란**: 대용량 라벨 없는 데이터로 언어모델을 먼저 학습하고, 그 위에 특정 작업용 신경망을 추가하는 방식. 상대적으로 적은 자원으로 여러 자연어 처리 작업을 수행할 수 있음

**양방향이 어려운 이유**
- 표준 언어모델은 다음 단어를 예측하는 과제 → 미래를 보면 안 됨 → 단방향 강제
- 양방향 Transformer를 그냥 쌓으면 각 토큰이 여러 층을 거치며 간접적으로 자기 자신을 볼 수 있음 → 예측이 무의미해짐

**BERT의 우회**
- 일부 토큰을 가리고 그것만 맞히게 함 → 나머지 전체 문맥을 양방향으로 써도 됨
- 제약은 과제에서 나온 것이지 모델에서 나온 게 아니었다는 뜻

**본 논문의 기여**
1. MLM으로 깊은 양방향 표현을 사전학습
1. 태스크마다 구조를 새로 설계하지 않고 출력 층만 바꿔 얹는 방식 확립
1. 11개 자연어 처리 태스크에서 당시 최고 성능

---

## 2. Related Work

2.1 Unsupervised Feature-based Approaches - ELMo 계열. 사전학습된 표현을 고정된 feature로 가져다 쓰고 태스크별 구조는 따로 설계

2.2 Unsupervised Fine-tuning Approaches - OpenAI GPT 계열. 사전학습 모델 전체를 fine-tuning. 다만 단방향(left-to-right)

**차별점**
- GPT는 단방향, ELMo는 좌→우와 우→좌를 따로 학습해 이어붙인 얕은 결합
- BERT는 처음부터 깊은 양방향 → 모든 층에서 양쪽 문맥이 함께 반영됨

---

## 3. BERT

### 입력 표현

**입력 요소** - 세 임베딩의 합. 셋 모두 학습 가능하다.

```python
self.token_embeddings    = nn.Embedding(vocab_size, hidden_size)
self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
self.segment_embeddings  = nn.Embedding(type_vocab_size, hidden_size)

embeddings = word_embeddings + pos_embeddings + seg_embeddings
embeddings = self.layer_norm(embeddings)
```

- **Token Embedding** - 토큰 ID를 임베딩 벡터로. `vocab_size` 는 tokenizer 어휘 사전 크기(`[CLS]`, `apple`, `##ing`, `[SEP]` …)
- **Segment Embedding** - `[SEP]` 으로 문장을 나누고 0/1로 표시. 두 문장까지만 쓸 수 있어 두 문장씩 잘라서 학습
- **Position Embedding** - 각 위치에 해당하는 고유 벡터. `max_position_embeddings` 가 한 번에 처리 가능한 최대 토큰 수

![그림](/img/bert/01.png)

### 3.1 Pre-training BERT

**NSP (Next Sentence Prediction)**
- 두 문장의 관계를 이해하기 위해, 두 번째 문장이 첫 문장의 바로 다음에 오는지 예측
- 50%는 실제로 이어지는 문장, 50%는 말뭉치에서 나온 임의의 문장
- `[CLS]` 토큰의 출력을 2차원 벡터로 변환하고 `IsNext` 라벨을 softmax로 할당

**MLM (Masked Language Model)**
- 전체 토큰의 15%를 마스킹하고, 그 안에서 다시 나눔

| 비율 | 처리 |
|---|---|
| 80% | 토큰을 `[MASK]` 로 바꿈 |
| 10% | 토큰을 무작위 단어로 바꿈 |
| 10% | 그대로 유지 |

**80/10/10으로 나누는 이유**
- `[MASK]` 는 사전학습에만 등장하고 파인튜닝 시점에는 나타나지 않음
- 100%를 `[MASK]` 로 바꾸면 모델이 "마스크 자리만 신경 쓰면 된다"고 학습해버림 → 실제 문장에서 표현이 약해짐
- 무작위 단어 10% → 어떤 토큰이든 문맥으로 검증하게 만듦
- 원본 유지 10% → 마스크 표시가 없어도 그 위치의 표현을 제대로 만들도록 강제

**전체 손실** - MLM과 NSP는 독립적인 태스크로 동시에 수행되며 각각 계산한 뒤 합산
```text
Loss_total = Loss_MLM + Loss_NSP
```

### 3.2 Fine-tuning BERT

- 최대 토큰 수만큼 끊어서 학습
- 사전학습된 파라미터 위에 태스크별 출력 층 하나만 얹으면 됨 → 구조를 크게 바꾸지 않고 다양한 태스크로 전이

---

## 4. Experiments

| 벤치마크 | 성능 |
|---|---|
| GLUE | 80.5% (기존 대비 +7.7%p) |
| SQuAD v1.1 | F1 93.2 |
| SQuAD v2.0 | F1 83.1 |
| MultiNLI | 86.7% |

- 11개 자연어 처리 태스크에서 당시 최고 성능
- 구조를 태스크마다 새로 설계하지 않고 같은 사전학습 모델에 출력 층만 바꿔 얹었다는 점이 함께 강조됨

---

## 5. Ablation Studies

- **5.1 사전학습 과제의 효과** - NSP를 빼거나 MLM 대신 단방향 LM을 쓰면 성능이 떨어짐. 양방향성이 핵심 기여임을 분리해 보인 실험
- **5.2 모델 크기의 효과** - 크기를 키울수록 좋아지고, 데이터가 적은 태스크에서도 그랬음
- **5.3 feature-based 접근** - fine-tuning 없이 고정 feature로 써도 경쟁력 있음

---

## 정리

BERT가 한 일은 "양방향으로 보면 안 된다"는 제약을 과제 설계로 우회한 것이다.

다음 단어를 맞히는 과제는 본질적으로 단방향을 요구한다. 그런데 일부를 가리고 그것만 맞히는 과제로 바꾸면 나머지 전체를 양방향으로 볼 수 있다. 제약은 과제에서 나온 것이지 모델에서 나온 게 아니었다는 뜻이다.

그리고 80/10/10 배분은 사전학습과 배포 사이의 분포 불일치를 다루는 사례다. 학습 시점에만 존재하는 인공물(`[MASK]`)에 모델이 의존하지 않게 만드는 것이고, 같은 문제는 다른 곳에서도 반복된다.

