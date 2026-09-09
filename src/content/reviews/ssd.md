---
title: SSD
paper: "SSD: Single Shot MultiBox Detector"
venue: ECCV 2016
link: https://arxiv.org/abs/1512.02325
claim: region proposal과 feature resampling 없이, 여러 해상도의 feature map에 default box를 깔아 한 번의 forward로 검출한다.
tags: [Object Detection, One-stage]
tier: main
date: 2025-03-05
draft: false
---

### **1. Introduction**

**기존 Object Detection 모델의 구조 및 한계:**
- 구조:
  (1) Bounding Box 를 생성→(2) pixels or features resampling→(3) High quality classifier
  > *[그림 자리 — Notion 원본에서 옮겨야 함]*

- 한계:
  Faster R-CNN은 (1), (2) 과정으로 인해 속도 느림, YOLO는 single-shot detection으로 속도를 높였으나 accuracy 부족하다
**SSD의 차별점:**
- Faster R-CNN과 달리 proposal, Feature resampling 없이 바로 detection 수행한다.
- YOLO와 같은 Single-shot detection이면서도 high accuracy 제공한다.

---

### **2. SSD (Single Shot MultiBox Detector)**

SSD는 하나의 CNN을 사용하여 고정된 개수의 Default Box를 기반으로 바운딩 박스를 조정하고, 해당 박스에서 객체가 존재할 확률을 예측하는 방식
**특이점:**
- 여러 크기의 Feature map을 활용한다.
- Default Boxes를 각 Feature map에서 활용한다
  - Default box:
    - 객체 탐지를 위해 한 pixel 기준 사전 생성하는 서로 다른 크기와 비율의 갖는 boxes
    - Feature map 통과 시 모든 픽셀마다 동일한 기준의 boxes생성
    - default box 기반 classification, bounding box regression
**구조:**
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

> *[그림 자리 — Notion 원본에서 옮겨야 함]*

1. 기존 VGG-16 네트워크를 Backbone 으로 사용하며 FC Layer를 제거한 후,  feature map을 더 다양하게 만들어 이 feature map들을 통해 classification 하게 된다. 이후 마지막 conv가 예측 내용을 취합해 최종 예측을 위한 진행한다.
1. 이후 feature map에서 객체를 탐지하기 위해 bounding box regression과 classification을 담당하는 head 들을 추가한다.
1. 각 Feature Map의 모든 픽셀에 있는 여러 개의 Default box를 Ground Truth와 비교하여 평가하고, 객체의 존재 여부와 box의 위치를 조정한다.
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

1. NMS를 적용하여 최종 탐지 결과를 결정.
  NMS의 동작 과정:
  - **모든 박스 중 Confidence Score 기준 정렬 후 가장 높은 박스를 선택**
    - 예: A(0.9), B(0.85), C(0.6) → A 선택
  - **선택한 박스(A)와 다른 박스들의 IOU를 계산(threshold 이상인 박스 제)**
    - A와 B의 IOU = 0.7 (기준 0.5 초과 → B 제거)
    - A와 C의 IOU = 0.3 (기준 0.5 미만 → C 유지)
  - **남아있는 박스들 중 가장 높은 Confidence Score를 가진 박스를 선택**
    - C(0.6) 선택
  - **C와 다른 박스들의 IOU를 계산하여 제거**
    - 남아있는 박스가 없으면 종료

#### **2.1 Model**

모델의 주요 구성 요소:
- **multi-scale feature maps 활용**: 추가적인 conv layers를 붙여, 점진적으로 축소되는 feature map을 생성하고, 이를 사용해 여러 크기의 객체를 검출한다.(feature map 크기 클 수록 작은 객체 탐지)
- **convolution 기반 예측 기법**: 각 feature map에 작은 3×3 conv filter를 적용하여 객체의 Predict category scores와 bounding box offset을 예측
- **Default Boxes와 다양한 종횡비(Aspect Ratios)**:
  - 여러 크기의 각 Feature map 셀마다 여러 개의 Default box를 설정하여 다양한 크기와 종횡비의 객체를 탐지.

#### **2.2 Training**

1. **기본 박스와의 매칭 (Matching Strategy)**
  - 모델 학습 시, 각 default box를 실제 object의 Bounding box와 매칭해야 함.
  - IoU 점수를 계산하여, 가장 높은 IoU를 가지는 default box를 실제 객체와 매칭.
  - IoU 0.5 이상인 경우 추가 매칭하여 학습을 좀 더 유연하게 만듦.
1. **손실 함수 (Loss Function)**
  - **Localization Loss (위치 오차)**: 실제 바운딩 박스와 예측한 바운딩 박스 간의 차이를 **Smooth L1 Loss** 를 사용하여 계산.
    - L2에 비해서 튀는 값을 잘 반영한다.(값을 예측해야하기 때문에 L2, L1같은 회귀 loss를 사용한다.)
  - **Confidence Loss (클래스 오차)**: 각 default box에서 예측한 class 확률과 실제 class 간 차이를 Softmax Loss 를 사용하여 계산.
  - 총 손실 함수:
    > *[그림 자리 — Notion 원본에서 옮겨야 함]*

    - N은 매칭된 기본 박스의 개수.
1. **Hard Negative Mining**
  - IoU ≥ 0.5 → Positive(객체 존재, 학습 대상), IoU < 0.5 → Negative(배경, 학습 대상 아님)로 판단.
  - Positive로 분류된 박스에 대해서만 Localization & Confidence Loss 적용. Negative 박스는 Classification Loss만 사용하여 학습.
    - 훈련 데이터에서 양성(positive) 샘플과 음성(negative) 샘플의 비율이 불균형하기 때문에, 음성 샘플 중 손실이 큰 상위 3배수만 학습.
1. **Data Augmentation**
  - 다양한 크기의 객체를 잘 탐지할 수 있도록 **random crop 및 다양한 비율의 사전 변형을 적용**.
  - 실험 결과, 데이터 증강을 추가하면 **mAP (Mean Average Precision) 성능이 8.8% 향상**됨.

---
