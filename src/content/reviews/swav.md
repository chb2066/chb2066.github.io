---
title: SwAV
paper: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
venue: NeurIPS 2020
link: https://arxiv.org/abs/2006.09882
claim: 인스턴스 비교 대신 prototype 할당을 서로 바꿔 예측하게 하면 메모리 뱅크 없이 대조 학습이 된다.
tags: [Self-supervised, Clustering]
tier: basic
date: 2025-06-04
draft: true
---

서론
SimCLR, MoCo 는 인스턴스 구별 방식을 사용, 모든 이미지를 다른 클래스로 구별하여 가까운 것끼리 가까워지게 먼 것끼리는 멀어지게 하는 방식을 사용했고 SimCLR은 메모리 사용량이 많았고, MOCO는 메모리 뱅크가 필요했다.
방법
1. SwAV는 클러스터링 기반 학습을 진행한다. 특징 벡터 대신 learnable vector를 만들어서 이미지가 어떤 prototype에 속하는 지 할당하는 코드를 계산한다.
1. clustering 을 통해서 discreate 한 분포가 아니라 연속적 분포를 얻어서 retrieval 이나 classification 작업에 바로 사용 가능함, 메모리 관리 효율 좋음
1. Swapping 메커니즘(동일 이미지에서 View A, View B 생성)
  1. 전역 View A의 특징을 통해 지역 View B가 어떤 클러스터에 속할 지 예측하고
  1. 지역 View B의 특징을 통해 전역 View A가 어떤 클러스터에 속할 지 예측하는 방식.
1. Multi-crop augmentation
  1. 기존 contrastive 방법들(SimCLR 등)은 이미지당 2개의 view(둘 다
고해상도 전체 crop)만 비교. SwAV는 여기에 **작은 해상도의
local crop을 여러 개 추가**로 사용.
    - 표준: 고해상도 global view 2개 + 저해상도 local view 여러 개
(예: 2×160px + 6×96px)
    - Local crop은 이미지의 일부(좁은 영역)만 담고 있지만 해상도가
작아서 연산 비용은 크게 안 늘어남
    - View 개수를 늘리면서도 **연산량 증가는 최소화** → "같은 배치
연산 비용으로 더 많은 정보/다양한 시점을 학습에 활용"하는 게
핵심 아이디어
    → Swapping 메커니즘과 결합: local view도 자기가 속한 prototype을
예측하게 학습시켜서, 작은 crop만 보고도 전체적인 semantic을
맞추도록 강제함 (일종의 local-to-global consistency)
1. 붕괴 방지
  1. Sinkhorn-Knopp 알고리즘을 사용
    1. 양수 행렬이 주어졌을 때, 이중 확률 행렬로 변환하는 것을 의미한다. “행의 합(가로 합)=1, 열의 합(세로 합)= 일정함”을 통해 특정 클러스터에 몰리지 못하게 되서 다양한 분류에 분포가 퍼지게 됨.
