---
title: CLIP
paper: Learning Transferable Visual Models From Natural Language Supervision
venue: ICML 2021
link: https://arxiv.org/abs/2103.00020
claim: 이미지-텍스트 쌍에 대한 contrastive learning만으로 zero-shot 전이가 가능한 시각 표현을 얻는다.
tags: [Vision-Language, Contrastive]
tier: main
date: 2025-04-10
draft: false
---

> 📄 [**Learning Transferable Visual Models From Natural Language Supervision**](https://arxiv.org/abs/2103.00020) · ICML 2021 · Radford, Kim, Hallacy et al.

## Abstract

**문제**
기존 시각 모델은 **미리 정해진 고정 클래스 집합**을 예측하도록 학습됨 → 새 개념을 다루려면 라벨을 새로 붙여야 하고, 일반성과 사용성이 제한됨

**해결책**
인터넷의 **4억 개 (이미지, 텍스트) 쌍**으로 "어떤 캡션이 어떤 이미지와 짝인지"를 맞히게 학습
자연어를 클래스 인터페이스로 써서 **학습 없이 분류 대상을 바꿈**

---

## 1. Introduction and Motivating Work

![Figure 1](/img/clip/f1.png)

**요지** - 자연어를 지도 신호로 써서, zero-shot 이미지 분류 성능을 **지도학습으로 훈련한 모델과 견줄 만한 수준**까지 올렸다.

**핵심 두 가지**
- 이미지에 대한 **보편적 개념**을 학습함
- **데이터 강건성이 뛰어남** → 환경이 바뀌면 성능이 급격히 떨어지는 기존 지도학습 모델과 달리 감소량이 매우 적음. Vision-Language 모델의 기본 특성

---

## 2. Approach

### 2.1 Natural Language Supervision

**아이디어** - 자연어에 포함된 지도 신호로부터 인식을 학습한다. 세부 라벨이 아니라 **자연어 자체를 훈련 신호로** 삼는다.

**자연어 지도의 강점**
- 자연어 지도를 확장하는 것이 image classification용 crowd-sourced labeling보다 쉬움
- 인터넷의 방대한 텍스트로부터 **수동적으로** 학습할 수 있음
- 단순히 표현을 학습하는 게 아니라 **그 표현을 언어와 연결** → 유연한 zero-shot 전이가 가능

### 2.2 Creating a Sufficiently Large Dataset

- 인터넷의 다양한 공개 소스에서 **4억 개의 (이미지, 텍스트) 쌍** 수집
- 광범위한 시각적 개념을 다루기 위해 **50만 개의 쿼리 집합** 중 하나를 포함하는 텍스트를 가진 쌍을 검색
- 쿼리당 최대 2만 개의 쌍을 넣어 **클래스 균형**을 맞춤

### 2.3 Selecting an Efficient Pre-Training Method

이 논문에서 가장 중요한 설계 결정이고, 근거가 실험으로 제시된다.

![Figure 2](/img/clip/f2.png)

**학습 목표의 세 단계 비교**
1. **정확한 캡션 단어를 예측** - 가장 직관적이지만 같은 이미지를 설명하는 방법은 무수히 많음 → **bag-of-words 기준선보다 3배 느리게** 학습
1. **bag-of-words 예측** - 어순을 버리고 어떤 단어가 등장하는지만 맞힘
1. **contrastive 목표** - 같은 기준선에서 예측을 대조로 바꾸면 **다시 4배의 효율 개선**

즉 **"정확한 단어가 무엇인가"를 버리고 "전체로서 어떤 텍스트가 어떤 이미지와 짝인가"만 남기는 것**이 핵심이다. 어려운 문제를 푸는 대신 쉬운 문제를 대규모로 푼다.

**입력 요소와 학습 방법**
- 이미지 인코더와 텍스트 인코더를 **공동으로 훈련**
- 각 인코더의 표현을 **multi-modal embedding space** 로 매핑하는 linear projection
- N개 쌍의 배치에서 N×N 조합 중 실제 쌍의 cosine similarity를 최대화하고 나머지는 최소화
- similarity score에 **symmetric cross entropy loss**
```text
symmetric CE loss = (loss_i + loss_t) / 2
```
텍스트→이미지, 이미지→텍스트 양방향으로 학습된다.

![Figure 3](/img/clip/f3.png)

**기타 설정**
- 사전 훈련 데이터 대부분이 단일 문장이므로 단일 문장 샘플링 함수를 제거
- augmentation 최소화 - random resize crop 정도만
- softmax의 logit 범위를 제어하는 temperature τ 를 수동 설정에서 **학습 가능하게** 변경

### 2.4 Choosing and Scaling a Model

**Image encoder - ResNet-50 계열 개조**
1. **antialiased rect-2 blur pooling** - 7×7 conv의 max pooling을 blur pooling으로. 부드럽게 만든 뒤 max pooling
1. **ResNet-D의 일부 구조** - stride=2 conv를 avg pooling으로 교체
1. 마지막 층의 **Global Average Pooling을 attention pooling으로 교체** - 평균으로 눌러 flatten하던 구조에서, 특정 부분에 attention하고 학습 가능한 파라미터로 최적화하는 구조로
1. EfficientNet의 아이디어로 채널 수·레이어 수·resolution을 **최적 비율로 함께** 키움

```python
# 기존
x = conv_blocks(x)           # [batch, 2048, 7, 7]
x = global_avg_pool(x)       # [batch, 2048]        7x7 을 하나로 압축
x = fully_connected(x)       # [batch, num_classes]

# 변경 후
x = conv_blocks(x)           # [batch, 2048, 7, 7]
x = attention_pooling(x)     # [batch, 2048]        GAP 대신 attention
x = projection_to_embed(x)   # [batch, embed_dim]
```

![그림](/img/clip/01.png)

**Image encoder - ViT**
- 기본 ViT 를 그대로 쓰되 patch·position embedding 앞에 layer normalization 추가, 초기화 방식만 가볍게 변경
- 여기도 EfficientNet 방식으로 스케일링

**Text encoder**
- Transformer. width만 ResNet 쪽 증가 비율에 맞춰 키움
- 텍스트 인코더 쪽 배율 조절은 **성능 향상에 거의 기여하지 않음** → width만 최소한으로 늘려 비율만 맞춤

---

## 3. Experiments

### 3.1.4 Prompt Engineering and Ensembling

zero-shot 성능을 실제로 끌어올린 요소이고, 논문이 별도 절로 다룬다.

![Figure 4](/img/clip/f4.png)

- **클래스 이름만 쓸 때의 문제** - `{label}` 하나만 넣으면 성능이 낮음. 학습 데이터의 텍스트는 대부분 문장인데 추론 시엔 단어 하나만 들어가 **분포가 어긋남**
- **기본 템플릿** - `"A photo of a {label}."` 이 좋은 기본값
- **도메인 정보 추가** - 반려동물 분류라면 `"A photo of a {label}, a type of pet."` 처럼 카테고리를 명시해 후보 공간을 좁힘
- **Ensembling** - `"A photo of a big {label}"`, `"A photo of a small {label}"` 같이 서로 다른 템플릿으로 만든 분류기를 결합

**둘을 함께 쓰면 zero-shot 성능이 유의미하게 오른다.** 모델을 바꾸지 않고 얻는 이득이다.

### 3.2 Representation Learning

![Figure 8](/img/clip/f8.png)

- zero-shot 성능과 linear probe 성능이 상관됨 → 표현 자체가 좋아서 zero-shot이 되는 것

### 3.3 Robustness to Natural Distribution Shift

- 분포가 바뀌었을 때 기존 ImageNet 모델보다 성능 감소가 훨씬 작음
- 고정 클래스에 맞춰 학습하지 않은 것이 강건성으로 이어짐

---

## 정리

이 논문에서 가져갈 것은 세 가지다.

**첫째, 목표를 쉽게 만들어 대규모로 미는 전략.** 정확한 캡션을 생성하는 어려운 문제 대신 짝을 맞히는 쉬운 문제로 바꿨더니 12배 효율이 나왔다. "무엇을 맞히게 할 것인가"가 데이터 규모만큼 중요하다.

**둘째, 자연어를 분류기 인터페이스로 만든 것.** 클래스 집합을 학습 없이 바꿀 수 있게 된다.

**셋째, 그 대가.** 캡션이 이미지의 주요 내용을 기술한다는 전제 위에 서 있으므로, **캡션에 잘 담기지 않는 축은 표현에서 사라진다.** 세밀한 속성, 공간 관계, 개수 같은 것들이다. 구성성이 약하다는 후속 지적이 여기서 나온다.

---

*`f`·`t` 로 시작하는 그림은 원 논문에서 가져왔다. Radford et al., [CLIP](https://arxiv.org/abs/2103.00020), ICML 2021.*
