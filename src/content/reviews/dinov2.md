---
title: DINOv2
paper: "DINOv2: Learning Robust Visual Features without Supervision"
venue: TMLR 2023
link: https://arxiv.org/abs/2304.07193
claim: 정제된 데이터로 retrieval 큐레이션한 대규모 데이터셋과 iBOT 기반 목적함수로, 파인튜닝 없이 쓸 수 있는 범용 시각 특징을 학습한다.
tags: [Self-supervised, Vision Backbone]
tier: main
date: 2025-06-10
draft: false
---

[DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

## Abstract

**문제**

자기지도 학습은 대규모일 때 데이터 품질 확보가 어려우며 다양성이 적다

**해결책**

- 정제 데이터를 검색 query 로 삼아 비정제 데이터에서 닮은 것만 골라내는 자동 큐레이션(LVD-142M)
- iBOT 기반 목적함수를 일부 수정하여, 큰 ViT model에서 소형 모델로 distillation
- 파인튜닝 없이 frozen feature 로 쓸 수 있는 높은 범용성을 가진다.

---

## 1. Introduction

**Self-supervised learning 방법론**
- Intra-image SSL -> 이미지의 가려진 부분을 복원. 표현은 나오지만 fine-tuning을 해야 쓸 수 있음
- Discriminative SSL -> 이미지 그룹 간 discreminative signal로 특징 학습. DINO, iBOT이 여기 속함

**Scaling의 문제**
- 데이터 양을 늘려 성능을 올린 연구들이 있었지만 데이터 품질이 낮다는 게 걸림돌
- 웹에서 긁어온 이미지는 소수의 지배적인 모드에 쏠려 있어 양이 늘어도 다양성이 따라오지 않음

---

## 3. Data Processing

**Image Retrieval로 데이터 정제**
- ImageNet 같은 정제 데이터로 특징을 뽑고, 인터넷 비정제 데이터로도 특징을 뽑은 뒤
- retrieval로 visual similarity를 재서 좋은 데이터와 분포가 비슷한 것만 학습 데이터에 추가

**검색 규칙은 query pool 크기에 따라 유동적**
- 기준 데이터가 크면 → 각 기준 이미지마다 가장 가까운 N개를 찾아옴
- 기준 데이터가 작으면 → 기준 이미지가 속한 클러스터에서 M개를 샘플링. 
- 유사도는 `q·k` 내적으로 구함

**데이터 구축 흐름 - LVD-142M**
1. 웹에서 `img` 태그 URL 추출. 부적절 URL·NSFW 필터링, 얼굴 블러를 거쳐 1.2B개 고유 이미지
1. **중복 제거** - copy detection 파이프라인으로 비정제 데이터 안의 중복을 먼저 없애고, 정제 데이터와 겹치는 것도 제거
1. ViT로 임베딩하고 이미지 간 유사도는 cosine similarity 로 측정
1. 비정제 데이터에 k-means clustering 적용 후 위 retrieval 규칙으로 선별

정제 데이터를 query로 사용한 것이 특징. 라벨 사용을 하지 않으면서도 기준잡힌 데이터 정제 가능
---

## 4. Discriminative Self-supervised Pre-training

iBOT loss와의 차이점.

**짧은 고해상도 단계**
- 낮은 해상도로 학습하다가 사전학습이 끝나기 직전 짧은 기간만 518×518로 올림
- 분할·검출 같은 픽셀 수준 태스크에서는 해상도가 중요(작은 물체가 저해상도에서 사라짐)
- 처음부터 고해상도로 학습하면 시간과 메모리가 너무 듦 → 말미에만 짧게

**head 분리**
- DINO loss와 iBOT loss가 각각 학습 가능한 MLP projection head를 씀
- 기존에는 두 loss가 head를 공유하는 게 낫다고 알려져 있었는데 규모를 키우면 반대임
- class token용과 patch token용 MLP head를 2개 따로 둬서 각 특징을 살림

**기존 DINO와 차이**
- **기존(DINO)** - softmax로 확률을 출력한 뒤 정답의 이동평균을 저장. centering으로 특정 칼럼이 정답으로 많이 나왔으면 점수를 깎고, sharpening으로 temperature 조절
- **DINOv2** - MLP로 logit을 뽑고, 배치 내 전체 이미지가 정답 칸에 골고루 들어가도록 SwAV의 Sinkhorn-Knopp 정규화 사용. 모든 칼럼을 각 칼럼 내 logit 합으로 나눠 특정 칼럼이 크면 패널티, 작으면 어드밴티지. 3회 반복

**KoLeo regularizer 추가**
- 각 특징 벡터가 개별적으로 구별되도록 만듦
- 최근접 이웃까지의 거리가 0이 되지 않게 log 거리 값을 최대화
- 표현이 한 점으로 뭉치는 것을 막는 장치

**teacher 특징 **
- 가장 큰 ViT-g를 먼저 학습시켜 고정한 뒤(EMA로 업데이트) 거기서 작은 모델들로 distill

---
## 6. Ablation Studies

- 6.1 위 다섯 가지 변경이 각각 얼마나 기여하는지 분해
- 6.4 KoLeo와 iBOT MIM 항을 빼면 성능이 떨어짐 → 둘 다 필요
- 6.5 distillation이 처음부터 학습시키는 것보다 나음
- 6.6 해상도 - 짧은 고해상도 단계만으로 dense task 성능을 얻음

