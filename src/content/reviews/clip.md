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

[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

## Abstract

**문제**
기존 Vision model은 미리 정해진 class를 예측하도록 학습됨 → 새 개념을 다루려면 라벨을 새로 붙여야 하고, 일반성과 사용성이 제한됨

**해결책**
인터넷의 4억 개 (이미지, 텍스트) 쌍으로 "어떤 캡션이 어떤 이미지와 짝인지"를 맞히게 학습

---

## 1. Introduction and Motivating Work

**요지** - 자연어를 지도 신호로 써서, zero-shot 이미지 분류 성능을 지도학습으로 훈련한 모델과 견줄 만한 수준까지 올림

**핵심 **
- 1. 이미지에 대한 보편적 개념을 학습함
- 2. 데이터 강건성이 뛰어남 → 환경이 바뀌면 성능이 급격히 떨어지는 기존 지도학습 모델과 달리 감소량이 매우 적음. Vision-Language 모델의 기본 특성

---

## 2. Approach

### 2.1 Natural Language Supervision

**아이디어** - 자연어에 포함된 supervised signal을 통한 학습을 진행한다. 세부 라벨이 아니라 자연어 자체를 training signal로 삼는다.

**자연어 지도의 강점**
- 자연어 지도를 확장하는 것이 image classification용 crowd-sourced labeling보다 쉬움
- 단순히 표현을 학습하는 게 아니라 그 표현을 언어와 연결 → 유연한 zero-shot 전이가 가능

### 2.2 Creating a Sufficiently Large Dataset

- 인터넷의 다양한 공개 소스에서 4억 개의 (이미지, 텍스트) 쌍 수집
- 광범위한 시각적 개념을 다루기 위해 50만 개의 쿼리 집합 중 하나를 포함하는 텍스트를 가진 쌍을 검색
- 쿼리당 최대 2만 개의 쌍을 넣어 클래스 균형을 맞춤

### 2.3 Selecting an Efficient Pre-Training Method

**입력 요소 및 학습 방법**
- 이미지 인코더와 텍스트 인코더를 둘 다 학습
- 각 인코더의 표현을 multi-modal embedding space 로 매핑하는 linear projection 사용
- 실제 쌍의 cosine similarity를 최대화하고 나머지는 최소화
- 텍스트→이미지, 이미지→텍스트 양방향 학습

**기타 설정**
- augmentation 최소화 - random resize crop 정도만
- softmax의 logit 범위를 제어하는 temperature τ 를 수동 설정에서 learable parameter로 변경

### 2.4 Choosing and Scaling a Model

**Image encoder - ResNet-50 계열 개조**
- ResNet-D의 일부 구조 차용 - stride=2 conv를 avg pooling으로 교체
- 마지막 층의 Global Average Pooling을 attention pooling으로 교체
- EfficientNet의 아이디어로 채널 수·레이어 수·resolution을 최적 비율로 함께 키움

**Image encoder - ViT**
- 기본 ViT 를 그대로 쓰되 patch·position embedding 앞에 layer normalization 추가
- EfficientNet 방식 스케일링

**Text encoder**
- Transformer. width만 ResNet 쪽 증가 비율에 맞춰 키움
- 텍스트 인코더 쪽 배율 조절은 성능 향상에 거의 기여하지 않음
  → width만 최소한으로 늘려 비율만 맞춤

## 3. Experiments

### 3.1.4 Prompt Engineering and Ensembling

zero-shot 성능을 실제로 끌어올린 요소

- **클래스 이름만 쓸 때의 문제** - `{label}` 하나만 넣으면 성능이 낮음. 학습 데이터의 텍스트는 대부분 문장인데 추론 시엔 단어 하나만 들어가 분포가 어긋남
- **기본 템플릿** - `"A photo of a {label}."` 이 기본값
- **도메인 정보 추가** - 반려동물 분류라면 `"A photo of a {label}, a type of pet."` 처럼 카테고리 명시
- **Ensembling** - `"A photo of a big {label}"`, `"A photo of a small {label}"` 같이 서로 다른 템플릿을 적용한 classifier ensemble

## 정리

- 이미지와 텍스트 쌍을 맞추게 하는 방식으로 일반화 성능을 높임

- 고정된 클래스 집합이 아닌 클래스를 학습 없이 사용 가능

- 텍스트에 표기되지 않은 표현들은 반영률이 낮아질 수 있음

---

Radford et al., [CLIP](https://arxiv.org/abs/2103.00020), ICML 2021.*
