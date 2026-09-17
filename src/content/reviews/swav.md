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

> 📄 [**Unsupervised Learning of Visual Features by Contrasting Cluster Assignments**](https://arxiv.org/abs/2006.09882) · NeurIPS 2020 · Caron, Misra, Mairal et al.

## Abstract

**문제**
SimCLR·MoCo 같은 **인스턴스 구별** 방식은 모든 이미지를 서로 다른 클래스로 보고 쌍별로 비교함
많은 negative sample 이 필요해 **큰 배치**(SimCLR)나 **메모리 뱅크**(MoCo)를 요구함

**해결책**
특징을 직접 비교하지 않고, 두 view 가 **서로의 클러스터 할당(코드)을 예측**하게 함 - swapped prediction
**Sinkhorn-Knopp** 의 균등 배분 제약으로 붕괴를 막고, 온라인·배치 단위로 할당을 계산
**multi-crop** 으로 연산량을 크게 늘리지 않으면서 관점 수를 늘림

---

## 1. Introduction

![Figure 1](/img/swav/f1.png)

**기존 상황**
- **SimCLR** - 메모리 사용량이 많음. 큰 배치가 필요하기 때문
- **MoCo** - 메모리 뱅크가 필요함
- 둘 다 **많은 negative sample 을 어떻게 확보할 것인가**의 문제를 서로 다른 방식으로 우회한 것

**클러스터링 기반으로 가면 얻는 것**
1. clustering 을 통해 이산 분포가 아니라 **연속적인 분포**를 얻으므로 retrieval·classification 에 바로 쓸 수 있음
1. **메모리 관리 효율이 좋음** - negative sample 을 저장할 필요가 없음

**본 논문의 기여**
1. 온라인 클러스터 할당을 쓰는 **swapped prediction** 목적함수 제안
1. 비용 대비 관점 수를 늘리는 **multi-crop** 전략 제안 - 다른 방법에도 적용 가능
1. ImageNet linear evaluation **75.3%**(ResNet-50)로 당시 SSL 최고 성능

---

## 3. Method

### 3.1 Online clustering

**입력 요소**
- 같은 이미지의 두 augmented view
- 공유 encoder 가 만든 특징 벡터
- **prototype** - 학습 가능한 벡터 집합. 이미지가 어떤 prototype 에 속하는지 나타내는 **코드**를 계산함

**Swapping 메커니즘** - 동일 이미지에서 View A 와 View B 를 만든 뒤, **서로의 코드를 예측**하게 한다.
- View A 의 특징으로 **View B 가 어떤 클러스터에 속할지** 예측
- View B 의 특징으로 **View A 가 어떤 클러스터에 속할지** 예측

왜 이렇게 하는가. 같은 이미지의 두 뷰는 **같은 클러스터에 속해야 한다**는 것이 넣고 싶은 제약이다. 그런데 이를 직접 강제하면 두 특징이 같아지는 자명한 해로 무너진다.

**서로의 할당을 예측하게** 만들면 제약이 간접적으로 걸린다. A 의 특징이 B 의 코드를 맞히려면 A 와 B 가 같은 의미를 담고 있어야 하지만, 특징 자체가 같을 필요는 없다.

**붕괴 방지 - Sinkhorn-Knopp**
- **문제** - 클러스터 할당을 학습하면 **모든 이미지가 하나의 클러스터로 몰리는** 자명한 해가 존재함
- **해결** - 양수 행렬을 **이중 확률 행렬**로 변환하는 Sinkhorn-Knopp 를 적용. "행의 합 = 1, 열의 합 = 일정"을 만족시키면 **특정 클러스터에 몰리지 못하게 되어** 분포가 여러 분류에 퍼짐

**온라인으로 푸는 것의 의미** - DeepCluster 같은 선행 연구는 전체 데이터셋에 대해 k-means 를 돌리는 오프라인 단계가 필요했다. SwAV 는 **배치 단위로 Sinkhorn 을 몇 번 반복**해서 할당을 구한다. 그래서 전체 데이터를 한 번 훑는 단계 없이 학습이 진행된다.

### 3.2 Multi-crop: Augmenting views with smaller images

**기존** - 이미지당 **2개의 view** 만 비교. 둘 다 고해상도 전체 crop

**SwAV** - 여기에 **작은 해상도의 local crop 을 여러 개 추가**
- 표준 구성 - 고해상도 global view 2개 + 저해상도 local view 여러 개 (예: 2×160px + 6×96px)
- local crop 은 이미지의 좁은 영역만 담지만 **해상도가 작아서 연산 비용이 크게 늘지 않음**
- **view 개수를 늘리면서도 연산량 증가는 최소화** → "같은 배치 연산 비용으로 더 많은 정보와 다양한 시점을 학습에 활용한다"가 핵심

**swapping 과 결합했을 때의 효과** - local view 도 자기가 속한 prototype 을 예측하게 학습시키면, **작은 crop 만 보고도 전체적인 semantic 을 맞히도록 강제**된다. 일종의 local-to-global consistency 다.

이 아이디어는 이후 DINO 계열이 그대로 가져간다.

---

## 4. Main Results

![Figure 2](/img/swav/f2.png)

- **4.1 ImageNet linear evaluation 75.3%**(ResNet-50) - 당시 자기지도 방법 중 최고 성능이고, 지도학습과의 격차를 크게 좁힘
- **4.2 전이** - 검출·분할 등 downstream 에서 지도학습 사전학습을 상회하는 항목이 나옴

![Table 3](/img/swav/t3.png)

- **4.3 작은 배치** - 배치 256 에서도 동작. 큰 배치나 메모리 뱅크가 필요 없음

---

## 5. Ablation Study

![Figure 3](/img/swav/f3.png)

- **5.1** 클러스터링 기반 설계의 각 요소 분해 - prototype 수, Sinkhorn 반복 횟수 등
- **5.2 multi-crop 은 다른 방법에도 이득을 줌** - SwAV 전용 장치가 아니라 일반적으로 적용 가능한 augmentation 전략임을 함께 보임
- **5.3** 더 오래 학습할수록 격차가 유지됨
- **5.4** 큐레이션되지 않은 10억 장 이미지로 사전학습해도 성능이 오름 → 웹 규모로 밀 수 있는 방법임

---

## 정리

SwAV에서 가져갈 재료는 세 가지고, 셋 다 이후 다른 방법으로 옮겨갔다.

**첫째, 쌍별 비교 대신 클러스터 할당을 비교하기.** negative sample을 정의하지 않아도 되고, 그래서 메모리 뱅크가 사라진다.

**둘째, Sinkhorn의 균등 배분 제약.** 이것 자체가 **떼어 쓸 수 있는 붕괴 방지 장치**다. DINOv2가 teacher 분포를 만들 때 centering 대신 이걸 가져다 쓴다.

**셋째, multi-crop.** 부분에서 전체를 예측하게 만드는 일반 레시피이고, 비용 대비 관점 수를 늘리는 방법이다.

가정도 함께 온다. **클러스터 수를 미리 정해야 하고, 균등 분포 가정이 실제 데이터와 다를 수 있다.** 롱테일 분포에서는 이 제약이 오히려 왜곡을 만든다.

---

*그림은 모두 원 논문에서 가져왔다. Caron et al., [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882), NeurIPS 2020.*
