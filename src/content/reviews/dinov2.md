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

관련 연구:
- Self-supervised learning
- Intra-image SSL
  - 이미지의 가려진 부분을 복원하는 방식
  - fine-tuning 해야 사용 가능함.
- Discriminative self-supervised learning
  - 이미지 그룹 간 판별 신호를 사용한 특징 학습
- Scaling self-supervised pretraining
  - 데이터 량을 늘려서 성능을 높인 연구들 존재
  - 데이터 품질이 낮다.
- Image Retrieval
  - Imagenet과 같은 정제된(Curated) 데이터를 사용해 특징들을 뽑고 인터넷(Uncurated) 데이터를 사용해 특징을 뽑은 뒤 Retireval을 통해 Visual Similarity를 사용하여 좋은 데이터와 분포가 비슷한 데이터만 학습 데이터에 추가하는 방식
    - 기존 데이터(Query)가 크면 각 기준 이미지마다 가장 가까운 N개의 이미지를 찾아오고, 기존 데이터가 작으면 기준 이미지가 속한 클러스터에서 M개를 샘플링한다.(소수 데이터를 많이 가져온다의 의미.)
      - q*k 간 내적을 사용해서 유사도를 구함.
데이터 제작 상세 과정:
1. uncurated data에서 curated data와 유사한 이미지들만 retrieval 해서 가져오는 방식 사용
1. 웹에서 img 태그의 url 추출
1. 중복제거
1. ViT 모델로 임베딩, 이미지 간 유사도는 Cosine-similarity를 사용,
기존과의 차이:
- Ibot의 loss를 그대로 차용하지만 일부 차이 존재
  - 작은 해상도로 학습하다가 종료 전 짧은 기간 동안 고해상도로 키워서 seg 성능 올림(그냥 고해상도는 비용이 비쌈)
  - MLP head 2개 생성해서 class 토큰이랑 patch 토큰 따로 적용시켜서 각 특징 살림(서로 영향 x, 독립적으로 작용)
  - Pt의 생성 방식에 차이점 존재
    - teacher의 feature extract 이후 projection head를 통해 점수 매기기 까지 동일
      - 기존: softmax로 확률 출력 후 정답의 평균을 저장, centering을 통해 특정 칼럼을 정답으로 많이 줬으면 점수를 깎고 sharpining을 통해 temperature를 조절 등
      - Dinov2: MLP 통과해서 Logit 뽑고 배치 내 전체 이미지가 정답 칸에 골고루 들어가게 SwAV 방식을 사용해서 특정 클래스에 안 쏠리게 보정함(모든 칼럼을 각 칼럼 내 logit 합으로 나눠서 특정 칼럼이 크면 패널티, 작으면 어드벤티지가 부여되는 구조로 만듬).
  - KoLeo Loss를 추가해서 각 특징 벡터들끼리 개별적으로 구별되게 함(NN을 이용해서 가장 가까운 이웃 벡터 거리가 0이 안되게 log 거리 값을 최대화함)
  - Teacher를 momentum encoder가 아니라 가장 큰 모델(ViT-g)을 먼저 학습시켜 고정한 뒤 distill
