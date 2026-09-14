---
title: SwAV
paper: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
venue: NeurIPS 2020
link: https://arxiv.org/abs/2006.09882
claim: 인스턴스 비교 대신 prototype 할당을 서로 바꿔 예측하게 하면 메모리 뱅크 없이 contrastive learning이 된다.
tags: [Self-supervised, Clustering]
tier: basic
date: 2025-06-04
draft: true
---

## 주요 전략
1. 특징 벡터를 직접 비교하지 않고 두 view가 서로의 클러스터 할당을 예측하는 swapped prediction을 수행함.
2. Sinkhorn-Knopp의 균등 배분 제약으로 collapse 방지, multi-crop으로 관점 수 확대함.

## 배경 지식

SimCLR과 MoCo는 **인스턴스 구별 방식**을 쓴다. 모든 이미지를 서로 다른 클래스로 보고, 가까운 것끼리는 가까워지게 먼 것끼리는 멀어지게 한다.

문제는 비용이다.

- **SimCLR** — 메모리 사용량이 많음. 큰 배치가 필요하기 때문임.
- **MoCo** — 메모리 뱅크가 필요함.

둘 다 **많은 negative sample을 어떻게 확보할 것인가**의 문제를 서로 다른 방식으로 우회한 것이다.

## SwAV method

### 클러스터링 기반 학습

SwAV는 특징 벡터를 직접 비교하지 않는다. 대신 **learnable vector**(prototype)를 만들어서, 이미지가 어떤 prototype에 속하는지 할당하는 **코드**를 계산한다.

이 방식의 이점은 두 가지다.

1. clustering을 통해 이산 분포가 아니라 **연속적인 분포**를 얻으므로, retrieval이나 classification 작업에 바로 쓸 수 있음.
2. **메모리 관리 효율이 좋음.** negative sample을 저장할 필요가 없음.

### Swapping 메커니즘

동일 이미지에서 View A와 View B를 만든 뒤, **서로의 코드를 예측**하게 한다.

- 전역 View A의 특징으로 **지역 View B가 어떤 클러스터에 속할지** 예측함.
- 지역 View B의 특징으로 **전역 View A가 어떤 클러스터에 속할지** 예측함.

왜 이렇게 하는가. 같은 이미지의 두 뷰는 **같은 클러스터에 속해야 한다**는 것이 우리가 넣고 싶은 제약이다. 그런데 이를 직접 강제하면 두 특징이 같아지는 자명한 해로 무너진다.

**서로의 할당을 예측하게** 만들면 제약이 간접적으로 걸린다. A의 특징이 B의 코드를 맞히려면 A와 B가 같은 의미를 담고 있어야 하지만, 특징 자체가 같을 필요는 없다.

## Multi-crop augmentation

기존 contrastive 방법들은 이미지당 **2개의 view**만 비교한다. 둘 다 고해상도 전체 crop이다.

SwAV는 여기에 **작은 해상도의 local crop을 여러 개 추가**한다.

- 표준 구성: 고해상도 global view 2개 + 저해상도 local view 여러 개 (예: 2×160px + 6×96px)
- local crop은 이미지의 좁은 영역만 담지만 **해상도가 작아서 연산 비용이 크게 늘지 않음.**
- **view 개수를 늘리면서도 연산량 증가는 최소화**함. "같은 배치 연산 비용으로 더 많은 정보와 다양한 시점을 학습에 활용한다"는 것이 핵심임.

### swapping과 결합했을 때의 효과

local view도 자기가 속한 prototype을 예측하게 학습시키면, **작은 crop만 보고도 전체적인 semantic을 맞히도록 강제**된다. 일종의 local-to-global consistency다.

이 아이디어는 이후 DINO 계열이 그대로 가져간다.

## 붕괴 방지 — Sinkhorn-Knopp

### 문제

클러스터 할당을 학습하면 **모든 이미지가 하나의 클러스터로 몰리는** 자명한 해가 존재한다.

### 해결

Sinkhorn-Knopp 알고리즘을 쓴다.

양수 행렬이 주어졌을 때 이를 **이중 확률 행렬**로 변환하는 것이다. "행의 합 = 1, 열의 합 = 일정함"을 만족시키면, **특정 클러스터에 몰리지 못하게 되어** 분포가 여러 분류에 퍼진다.

### 온라인으로 푸는 것의 의미

DeepCluster 같은 선행 연구는 전체 데이터셋에 대해 k-means를 돌리는 오프라인 단계가 필요했다. SwAV는 **배치 단위로 Sinkhorn을 몇 번 반복**해서 할당을 구한다. 그래서 전체 데이터를 한 번 훑는 단계 없이 학습이 진행된다.

## 실험에서 확인된 것

- **ImageNet linear evaluation 75.3%** (ResNet-50). 당시 자기지도 방법 중 최고 성능이고, 지도학습과의 격차를 크게 좁혔음.
- **multi-crop은 다른 방법에도 이득을 줌.** SwAV 전용 장치가 아니라 일반적으로 적용 가능한 augmentation 전략이라는 것을 함께 보였음.
- 작은 배치에서도 동작함. 큰 배치나 메모리 뱅크가 필요 없음.

## 정리

SwAV에서 가져갈 재료는 세 가지고, 셋 다 이후 다른 방법으로 옮겨갔다.

**첫째, 쌍별 비교 대신 클러스터 할당을 비교하기.** negative sample을 정의하지 않아도 되고, 그래서 메모리 뱅크가 사라진다.

**둘째, Sinkhorn의 균등 배분 제약.** 이것 자체가 **떼어 쓸 수 있는 붕괴 방지 장치**다. DINOv2가 teacher 분포를 만들 때 centering 대신 이걸 가져다 쓴다.

**셋째, multi-crop.** 부분에서 전체를 예측하게 만드는 일반 레시피이고, 비용 대비 관점 수를 늘리는 방법이다.

가정도 함께 온다. **클러스터 수를 미리 정해야 하고, 균등 분포 가정이 실제 데이터와 다를 수 있다.** 롱테일 분포에서는 이 제약이 오히려 왜곡을 만든다.
