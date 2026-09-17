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

> 📄 [**DINOv2: Learning Robust Visual Features without Supervision**](https://arxiv.org/abs/2304.07193) · TMLR 2023 · Oquab, Darcet, Moutakanni et al.

## Abstract

**문제**
자기지도 학습은 라벨 없이 표현을 얻을 수 있지만, 규모를 키울 때 **데이터 품질**이 걸림돌. 웹에서 긁은 이미지는 소수의 지배적 모드에 쏠려 양이 늘어도 다양성이 따라오지 않음

**해결책**
정제 데이터를 **검색 query** 로 삼아 비정제 데이터에서 닮은 것만 골라내는 자동 큐레이션(LVD-142M)
iBOT 기반 목적함수에 다섯 가지를 손보고, 큰 ViT에서 소형 모델로 distillation
**파인튜닝 없이 frozen feature 로 쓸 수 있는** 범용 특징 달성

---

## 1. Introduction

![Figure 1](/img/dinov2/f1.png)

**Self-supervised learning의 두 갈래**
- **Intra-image SSL** - 이미지의 가려진 부분을 복원. 표현은 나오지만 **fine-tuning을 해야 쓸 수 있음**
- **Discriminative SSL** - 이미지 그룹 간 판별 신호로 특징을 학습. DINO, iBOT이 여기 속함

**Scaling의 문제**
- 데이터 양을 늘려 성능을 올린 연구들이 있었지만 **데이터 품질이 낮다**는 게 걸림돌
- 웹에서 긁어온 이미지는 소수의 지배적인 모드에 쏠려 있어 **양이 늘어도 다양성이 따라오지 않음**

---

## 3. Data Processing

![Figure 3](/img/dinov2/f3.png)

**Image Retrieval을 큐레이션에 쓰기**
- ImageNet 같은 정제 데이터로 특징을 뽑고, 인터넷 비정제 데이터로도 특징을 뽑은 뒤
- retrieval로 visual similarity를 재서 **좋은 데이터와 분포가 비슷한 것만 학습 데이터에 추가**

**검색 규칙은 query pool 크기에 따라 달라진다**
- 기준 데이터가 크면 → 각 기준 이미지마다 가장 가까운 **N개**를 찾아옴
- 기준 데이터가 작으면 → 기준 이미지가 속한 **클러스터에서 M개**를 샘플링. 소수 데이터를 많이 가져온다는 뜻
- 유사도는 `q·k` 내적으로 구함

**데이터 구축 흐름 - LVD-142M**
1. 웹에서 `img` 태그 URL 추출. 부적절 URL·NSFW 필터링, 얼굴 블러를 거쳐 **1.2B개 고유 이미지**
1. **중복 제거** - copy detection 파이프라인으로 비정제 데이터 안의 중복을 먼저 없애고, 정제 데이터와 겹치는 것도 제거
1. ViT로 임베딩하고 이미지 간 유사도는 **cosine similarity** 로 측정
1. 비정제 데이터에 **k-means clustering** 적용 후 위 retrieval 규칙으로 선별

결과가 **LVD-142M**, 1억 4200만 장이다. 검색과 중복 제거는 Faiss로 처리한다.

여기서 눈여겨볼 점은 **정제 데이터가 라벨이 아니라 query로 쓰인다**는 것이다. 라벨을 쓰지 않으면서도 "무엇이 좋은 데이터인가"의 기준을 정제 데이터가 대신 정해준다.

---

## 4. Discriminative Self-supervised Pre-training

iBOT loss를 그대로 차용하되 **다섯 가지가 다르다.**

**① 짧은 고해상도 단계**
- 낮은 해상도로 학습하다가 **사전학습이 끝나기 직전 짧은 기간만 518×518로** 올림
- 분할·검출 같은 픽셀 수준 태스크에서는 해상도가 중요(작은 물체가 저해상도에서 사라짐)
- 처음부터 고해상도로 학습하면 시간과 메모리가 너무 듦 → 말미에만 짧게

**② head 분리**
- DINO loss와 iBOT loss가 각각 학습 가능한 MLP projection head를 씀
- 기존에는 두 loss가 head를 공유하는 게 낫다고 알려져 있었는데 **규모를 키우면 반대가 참**
- class token용과 patch token용 **MLP head를 2개 따로** 둬서 각 특징을 살림

**③ teacher 확률 분포의 생성 방식**
- **기존(DINO)** - softmax로 확률을 출력한 뒤 정답의 이동평균을 저장. centering으로 특정 칼럼이 정답으로 많이 나왔으면 점수를 깎고, sharpening으로 temperature 조절
- **DINOv2** - MLP로 logit을 뽑고, 배치 내 전체 이미지가 정답 칸에 골고루 들어가도록 **SwAV의 Sinkhorn-Knopp 정규화** 사용. 모든 칼럼을 각 칼럼 내 logit 합으로 나눠 특정 칼럼이 크면 패널티, 작으면 어드밴티지. 3회 반복

**④ KoLeo regularizer 추가**
- 각 특징 벡터가 개별적으로 구별되도록 만듦
- 최근접 이웃까지의 거리가 0이 되지 않게 log 거리 값을 최대화
- 표현이 한 점으로 뭉치는 것을 막는 장치

**⑤ teacher를 momentum encoder로 두지 않음**
- 가장 큰 **ViT-g(1B)를 먼저 학습시켜 고정**한 뒤 거기서 작은 모델들로 distill

![Figure 5](/img/dinov2/f5.png)

목적함수 자체가 이미 teacher→student distillation 형태이므로 **같은 학습 루프를 쓰되 몇 가지만 바꾼다.**
- 더 큰 모델을 **frozen teacher** 로
- student의 EMA를 따로 유지해 **그것을 최종 모델로**
- masking과 stochastic depth **제거**
- iBOT loss는 **두 개의 global crop에만** 적용

ablation에서 이 방식이 처음부터 학습시키는 것보다 낫고, **ViT-L 크기에서도 그랬다.**

---

## 5. Efficient Implementation

대규모로 밀어붙일 수 있었던 이유가 여기 있다. A100 GPU에서 PyTorch 2.0으로 학습한다.

- **FlashAttention 자체 구현** - 메모리 효율적인 attention을 직접 구현
- **Sequence packing** - DINO는 224 해상도 large crop과 작은 local crop을 함께 forward해야 함. 길이가 다른 시퀀스를 따로 돌리면 낭비 → NLP에서 온 sequence packing으로 **여러 시퀀스를 이어붙여 한 번에 처리**
- **Efficient stochastic depth** - 건너뛸 층의 연산을 실제로 생략
- **FSDP** - AdamW 옵티마이저 상태를 여러 GPU에 분산 → **모델 크기가 한 GPU 메모리에 묶이지 않음**. GPU 간 통신 비용도 감소

---

## 6. Ablation Studies

![Table 1](/img/dinov2/t1.png)

- **6.1** 위 다섯 가지 변경이 각각 얼마나 기여하는지 분해
- **6.4** KoLeo와 iBOT MIM 항을 빼면 성능이 떨어짐 → 둘 다 필요
- **6.5** distillation이 처음부터 학습시키는 것보다 나음
- **6.6** 해상도 - 짧은 고해상도 단계만으로 dense task 성능을 얻음

---

## 7. Results

![Table 4](/img/dinov2/t4.png)

- **7.1 ImageNet 분류** - frozen feature에 linear probe만으로 높은 성능
- **7.2 기타 이미지·비디오 분류** - 세밀한 분류 벤치마크에서도 일관됨
- **7.3 Instance recognition** - 검색 태스크에서도 강함

파인튜닝 없이 **얼린 특징 그대로** 여러 태스크를 커버한다는 것이 이 논문의 주장이다.

---

## 정리

이 논문에서 가져갈 만한 것은 두 가지다.

**첫째, 정제 데이터를 라벨이 아니라 검색 query로 쓰는 발상.** "무엇이 좋은 데이터인가"를 사람이 라벨링하는 대신, 이미 좋다고 알려진 데이터와의 유사도로 정의한다. 라벨 없이 큐레이션이 가능해진다.

**둘째, 규모가 바뀌면 최적 설계가 뒤집힌다는 관찰.** head 공유가 작은 규모에서는 낫지만 큰 규모에서는 분리가 낫다는 것을 실제로 확인했다. 작은 실험에서 얻은 설계 결정을 그대로 스케일업하면 안 된다는 뜻이다.

---

*그림은 모두 원 논문에서 가져왔다. Oquab et al., [DINOv2](https://arxiv.org/abs/2304.07193), TMLR 2023.*
