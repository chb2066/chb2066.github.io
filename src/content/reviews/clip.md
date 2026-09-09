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

### instruction

자연어를 사용한 방법으로 zeroshot 기반 이미지 분류 등에서 성능을, 학습한 모델의 성능과 견줄만큼 상승시켰다.

#### 핵심

- 이미지에 대한 보편적 개념을 학습함
  - 학습 모델에 비해서 데이터 강건성이 뛰어나기 때문에 환경 변화 시 성능이 급격히 감소하는 기존 학습 모델들에 비해 성능 감소량이 매우 적음.(Vision-Language model의 기본 특성)

### Approach

#### 2.1. Natural Language Supervision

**아이디어:**
자연어에 포함된 지도학습으로부터 인식을 학습한다
**강조점:**
지도 학습을 통해 세부 사항이 아니라 자연어를 훈련 신호로 인식한다
**자연어의 학습의 강점:**
- 자연어 지도 학습을 확장하는 게 image classification 을 위한 crowd-sourced labeling 보다 쉽다.
- 인터넷 상의 방대한 양의 텍스트에 포함된 지도 학습으로부터 수동적으로 학습할 수 있습니다. "단순히" 표현을 학습하는 것이 아니라 그 표현을 언어와 연결하여 유연한 제로샷 전이를 가능하게 한다.

#### 2.2.  Creating a Sufficiently Large Dataset

- 인터넷의 다양한 공개 소스에서 수집한 4억 개의 (이미지, 텍스트) 쌍으로 구성된 새로운 데이터셋을 구축
- 광범위한 시각적 개념을 다루기 위해 50만 개의 쿼리 세트 에 있는 쿼리 중 하나를 포함하는 텍스트를 가진 (이미지, 텍스트) 쌍을 검색
- 쿼리당 최대 20,000개의 (이미지, 텍스트) 쌍을 포함하여 클래스 균형 맞춤.

#### 2.3. Selecting an Efficient Pre-Training Method

**학습 이유:**
대조적 목표가 동등한 예측 목표보다 더 나은 표현을 학습할 수 있다
**훈련 방향성:**
텍스트의 정확한 단어가 아니라 전체로서 어떤 텍스트가 어떤 이미지와 쌍을 이루는지만 예측 작업을 해결하는 시스템을 훈련
**훈련 방법:**
- CLIP는 이미지 인코더와 텍스트 인코더를 공동으로 훈련
  - 각 인코더 표현을 multi-modal embedding space으로 매핑하기 위해 linear projection(벡터 투영정도의 개념)을 사용
- N개의 (이미지, 텍스트) 쌍 배치가 주어지면 N*N 개의 쌍 중 실제 쌍에 대한 cosine similarity 를 최대화하고 잘못된 쌍에 대한 cosine similarity를 최소화하는 방향으로 임베딩 공간을 학습한다.
  - ex [1. 이미지1, 텍스트1. 2. 이미지1, 텍스트2] 중  [1. 이미지1, 텍스트1.]
  - similarity score에 대해 symmetric cross entropy loss 를 optimize한다. symmetric CE loss = (loss_i + loss_t) / 2
    - 텍스트→이미지, 이미지→텍스트 양방향 학습 가능
기타 사항:
- 사전 훈련 데이터셋의 대부분의 쌍이 단일 문장이다. 따라서 단일 문장 샘플링 함수를 제거한다.
- aug 최소화하여 random resize crop 정도만 사용한다.
- 마지막으로, 소프트맥스에서 로짓 범위를 제어하는  매개변수 τ는 기존 수동에서 자동으로 변경.

#### 2.4. Choosing and Scaling a Model

**Image encoder**
1. Renset50
  1. antialiased rect-2 블러 풀링(blur pooling) 적용
    1. 77conv의 기존 maxpool을 Blur Pooling로 변경(부드럽게 바꾸고 maxpool)
  1. Resnet-D의 일부를 바꾼 구조를 사용한다.(Resnet-D 설명(bag off trics 논문 속 구조): conv의 stride=2를 avgpool로 바꿔서 성능 향상)
  1. 마지막 layer에 있던 Global avg pooling을 Attention pooling으로 바꿈. 이를 통해 단순히 평균 값으로 flat하던 구조에서 특정 부분에 attention하고 학습하는 파라미터를 통해 값을 더 최적화 하는 구조로 바뀜.
  1. efficient-net의 아이디어를 통해 채널 수, 레이어 수, resolution을 최적의 성능을 내는 비율로 증가시킴.

```python
#기존
x = conv_blocks(x)           # [batch, 2048, 7, 7] (ResNet-50 기준)
x = global_avg_pool(x)       # [batch, 2048] ← 7x7을 하나씩으로 압축
x = fully_connected(x)       # [batch, num_classes]
```

```python
#변경 후
x = conv_blocks(x)           # [batch, 2048, 7, 7]
x = attention_pooling(x)     # [batch, 2048] ← GAP 대신 attention 사용
x = projection_to_embed(x)   # [batch, embed_dim]
```

![그림 1](/img/clip/01.png)

![그림 2](/img/clip/02.png)

1. ViT:
  1. 기본 ViT 논문에 있는 내용 그대로 사용. patch and position embeddings 전에 layer normalization을 추가하고 가벼운 초기화 방식으로 변경
  1. efficient-net의 아이디어를 통해 채널 수, 레이어 수, resolution을 최적의 성능을 내는 비율로 증가 시킴.
**Text encoder**
Transformer 사용.
- width만 efficient-net의 구조에 의해 변형된 Resnet의 증가 비율만큼 증가 시킴.
  - CLIP의 성능 관점에서 efficient net의 배율 조절은 성능 향상에 도움을 거의 주지 못하기에 width만 최소한으로 증가 시켜서 Image-encoder와 비율만 맞춤.
