<!--
개정: 2026-09-10 (원본: src/content/reviews/clip.md)
- 절 번호(2.1, 2.4 …)를 내용 제목으로 교체하고 i-jepa 형식으로 재배치
- 「왜 contrastive 인가」 신설 — 원본은 "대조적 목표가 더 낫다"고만 적었는데,
  논문 Figure 2 의 근거를 채웠다. 정확한 단어를 예측하는 목표는 bag-of-words 기준선보다
  3배 느리고, 이를 contrastive 로 바꾸면 다시 4배 빨라진다
- 「Prompt engineering 과 ensembling」 신설 (논문 3.1.4). 원본에 완전히 빠져 있었는데
  zero-shot 성능을 실제로 끌어올린 요소라 빼면 그림이 안 맞는다
- 끝맺음을 평서형으로 통일
- 원본의 인코더 개조 설명(blur pooling, ResNet-D, attention pooling)과 코드 비교는
  그대로 유지. 이 글에서 가장 구체적인 부분이다
-->
---
title: CLIP
paper: Learning Transferable Visual Models From Natural Language Supervision
venue: ICML 2021
link: https://arxiv.org/abs/2103.00020
claim: 이미지-텍스트 쌍에 대한 대조 학습만으로 zero-shot 전이가 가능한 시각 표현을 얻는다.
tags: [Vision-Language, Contrastive]
tier: main
date: 2025-04-10
draft: false
---

핵심 키워드:
natural language supervision, contrastive pretraining, zero-shot transfer, 4억 쌍 데이터셋
주요 전략:
정확한 단어를 맞히는 대신 "어떤 텍스트가 어떤 이미지와 짝인지"만 맞히게 해서 학습 효율을 끌어올린다
사용 가능 분야:
zero-shot 분류, 검색, 다른 모델의 텍스트 정렬 인코더

#### 요지

자연어를 지도 신호로 써서, zero-shot 이미지 분류 성능을 **지도학습으로 훈련한 모델과 견줄 만한 수준**까지 올렸다.

**핵심**
- 이미지에 대한 보편적 개념을 학습한다.
- 데이터 강건성이 뛰어나다. 환경이 바뀌면 성능이 급격히 떨어지는 기존 지도학습 모델과 달리, 성능 감소량이 매우 적다. Vision-Language 모델의 기본 특성이다.

#### 배경 지식 — Natural Language Supervision

**아이디어**
자연어에 포함된 지도 신호로부터 인식을 학습한다.

**강조점**
세부 라벨이 아니라 **자연어 자체를 훈련 신호로** 삼는다.

**자연어 지도의 강점**
- 자연어 지도를 확장하는 것이 image classification용 crowd-sourced labeling보다 쉽다.
- 인터넷에 있는 방대한 텍스트로부터 **수동적으로** 학습할 수 있다.
- 단순히 표현을 학습하는 것이 아니라 **그 표현을 언어와 연결**한다. 그래서 유연한 zero-shot 전이가 가능해진다.

#### 데이터셋 구축

- 인터넷의 다양한 공개 소스에서 수집한 **4억 개의 (이미지, 텍스트) 쌍**으로 새 데이터셋을 만든다.
- 광범위한 시각적 개념을 다루기 위해, **50만 개의 쿼리 집합** 중 하나를 포함하는 텍스트를 가진 쌍을 검색한다.
- 쿼리당 최대 2만 개의 쌍을 넣어 **클래스 균형**을 맞춘다.

#### 왜 contrastive 인가

이 논문에서 가장 중요한 설계 결정이고, 근거가 실험으로 제시된다.

**학습 목표를 세 단계로 비교한다.**

1. **이미지의 정확한 캡션 단어를 예측하기** — 가장 직관적인 목표다. 그런데 같은 이미지를 설명하는 방법은 무수히 많다. 이 목표는 **bag-of-words 인코딩 기준선보다 3배 느리게** 학습한다.
2. **bag-of-words 예측** — 어순을 버리고 어떤 단어들이 등장하는지만 맞힌다.
3. **contrastive 목표** — 같은 bag-of-words 기준선에서 예측 목표를 대조 목표로 바꾸면 **다시 4배의 효율 개선**이 나온다.

즉 **"정확한 단어가 무엇인가"를 버리고 "전체로서 어떤 텍스트가 어떤 이미지와 짝인가"만 남기는 것**이 핵심이다. 어려운 문제를 푸는 대신 쉬운 문제를 대규모로 푼다.

**학습 방법**
- 이미지 인코더와 텍스트 인코더를 **공동으로 훈련**한다.
- 각 인코더의 표현을 **multi-modal embedding space**로 매핑하기 위해 linear projection을 쓴다.
- N개의 (이미지, 텍스트) 쌍 배치가 주어지면, N×N 조합 중 실제 쌍의 cosine similarity를 최대화하고 나머지는 최소화한다.
- similarity score에 대해 **symmetric cross entropy loss**를 최적화한다.

```text
symmetric CE loss = (loss_i + loss_t) / 2
```

텍스트→이미지, 이미지→텍스트 양방향으로 학습된다.

**기타 설정**
- 사전 훈련 데이터셋의 대부분이 단일 문장이므로 단일 문장 샘플링 함수를 제거한다.
- augmentation을 최소화한다. random resize crop 정도만 쓴다.
- softmax의 logit 범위를 제어하는 temperature 파라미터 τ를 수동 설정에서 **학습 가능하게** 바꾼다.

#### 모델 선택과 스케일링

**Image encoder — ResNet-50 계열 개조**

1. **antialiased rect-2 blur pooling** 적용. 7×7 conv의 기존 max pooling을 blur pooling으로 바꾼다. 부드럽게 만든 뒤 max pooling한다.
2. **ResNet-D의 일부 구조**를 가져온다. stride=2인 conv를 avg pooling으로 바꿔 성능을 올리는 방식이다.
3. 마지막 층의 **Global Average Pooling을 attention pooling으로 교체**한다. 단순히 평균으로 눌러 flatten하던 구조에서, 특정 부분에 attention하고 학습 가능한 파라미터로 값을 최적화하는 구조로 바뀐다.
4. EfficientNet의 아이디어를 가져와 채널 수, 레이어 수, resolution을 **최적 비율로 함께** 키운다.

```python
# 기존
x = conv_blocks(x)           # [batch, 2048, 7, 7]  (ResNet-50 기준)
x = global_avg_pool(x)       # [batch, 2048]        7x7 을 하나로 압축
x = fully_connected(x)       # [batch, num_classes]
```

```python
# 변경 후
x = conv_blocks(x)           # [batch, 2048, 7, 7]
x = attention_pooling(x)     # [batch, 2048]        GAP 대신 attention
x = projection_to_embed(x)   # [batch, embed_dim]
```

![그림 1](/img/clip/01.png)

![그림 2](/img/clip/02.png)

**Image encoder — ViT**
- 기본 ViT 논문의 내용을 그대로 쓴다. patch embedding과 position embedding 앞에 layer normalization을 추가하고 초기화 방식을 가볍게 바꾼 정도다.
- 여기도 EfficientNet의 아이디어로 채널 수, 레이어 수, resolution을 최적 비율로 키운다.

**Text encoder**
Transformer를 쓴다. width만 ResNet 쪽의 증가 비율에 맞춰 키운다.

CLIP의 성능 관점에서 텍스트 인코더 쪽의 배율 조절은 **성능 향상에 거의 기여하지 않는다.** 그래서 width만 최소한으로 늘려 image encoder와 비율만 맞춘다.

#### Prompt engineering 과 ensembling

zero-shot 성능을 실제로 끌어올린 요소이고, 논문이 별도 절로 다룬다.

**클래스 이름만 쓰면 안 된다**
`{label}` 하나만 텍스트 인코더에 넣으면 성능이 낮다. 학습 데이터의 텍스트는 대부분 문장이었는데 추론 시에는 단어 하나만 들어가서 분포가 어긋나기 때문이다.

**기본 템플릿**
`"A photo of a {label}."` 이 좋은 기본값이다.

**도메인 정보를 넣으면 더 좋아진다**
반려동물 분류라면 `"A photo of a {label}, a type of pet."` 처럼 카테고리를 명시한다. 후보 공간을 좁혀주는 효과가 있다.

**Ensembling**
여러 zero-shot 분류기를 앙상블한다. `"A photo of a big {label}"`, `"A photo of a small {label}"` 같이 서로 다른 템플릿으로 만든 분류기들을 결합한다.

**prompt engineering과 ensembling을 함께 쓰면 zero-shot 성능이 유의미하게 오른다.** 모델을 바꾸지 않고 얻는 이득이다.

#### 정리

이 논문에서 가져갈 것은 세 가지다.

**첫째, 목표를 쉽게 만들어 대규모로 미는 전략.** 정확한 캡션을 생성하는 어려운 문제 대신 짝을 맞히는 쉬운 문제로 바꿨더니 12배 효율이 나왔다. "무엇을 맞히게 할 것인가"가 데이터 규모만큼 중요하다.

**둘째, 자연어를 분류기 인터페이스로 만든 것.** 클래스 집합을 학습 없이 바꿀 수 있게 된다.

**셋째, 그 대가.** 캡션이 이미지의 주요 내용을 기술한다는 전제 위에 서 있으므로, **캡션에 잘 담기지 않는 축은 표현에서 사라진다.** 세밀한 속성, 공간 관계, 개수 같은 것들이다. 구성성이 약하다는 후속 지적이 여기서 나온다.
