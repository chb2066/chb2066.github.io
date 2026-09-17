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

> 📄 [**SSD: Single Shot MultiBox Detector**](https://arxiv.org/abs/1512.02325) · ECCV 2016 · Liu, Anguelov, Erhan et al.

## Abstract

**문제**
기존 검출기는 box 후보를 만들고 픽셀·feature를 resampling한 뒤 분류기에 넣는 세 단계 구조 → 정확하지만 느림
속도를 높인 single-shot 방식(YOLO)은 정확도가 부족

**해결책**
proposal과 resampling 단계를 **아예 제거**하고, 여러 해상도의 feature map 각 위치에 **default box를 미리 깔아** 한 번의 forward로 검출
종횡비별로 예측을 분리해 속도를 얻으면서 정확도를 지킴

---

## 1. Introduction

**기존 검출 모델의 구조**

![Fig. 1](/img/ssd/f1.png)

1. bounding box를 생성
1. 픽셀이나 feature를 resampling
1. 고품질 분류기에 넣음

**여기서 문제가 갈린다**
- **Faster R-CNN** - (1)과 (2) 때문에 느림
- **YOLO** - single-shot으로 속도를 높였지만 정확도가 부족

**SSD의 차별점**
- Faster R-CNN과 달리 **proposal도 feature resampling도 없이** 바로 검출
- YOLO처럼 single-shot이면서 **높은 정확도**

속도와 정확도를 동시에 얻는 방법이 이 논문의 내용이다.

---

## 2. The Single Shot Detector (SSD)

하나의 CNN으로 **고정된 개수의 default box를 기반**으로 바운딩 박스를 조정하고, 각 박스에서 객체가 존재할 확률을 예측한다.

**두 가지 특징**
1. **여러 크기의 feature map을 활용**
1. **각 feature map에서 default box를 사용**

**Default box란**
- 객체 탐지를 위해 한 픽셀 기준으로 미리 생성하는, 서로 다른 크기와 비율의 박스들
- feature map을 통과할 때 **모든 픽셀마다 동일한 기준의 박스들**이 생성됨
- 이 박스를 기준으로 classification과 bounding box regression 수행

### 2.1 Model

![Fig. 2](/img/ssd/f2.png)

**입력 요소와 흐름**
1. **VGG-16을 백본**으로 쓰되 **FC Layer 제거**. 그 뒤 feature map을 더 다양하게 만들고, 이들로 classification 수행. 마지막 conv가 예측을 취합
1. feature map에서 객체를 탐지하기 위해 **bounding box regression과 classification을 담당하는 head** 추가
1. 각 feature map의 모든 픽셀에 있는 default box를 ground truth와 비교해 평가하고, 객체 존재 여부와 박스 위치 조정
1. **NMS**를 적용해 최종 탐지 결과 결정

![그림](/img/ssd/04.png)

**주요 구성 요소 세 가지**
- **다중 스케일 feature map** - 추가 conv layer로 점진적으로 축소되는 feature map을 만들어 여러 크기의 객체를 검출. **feature map이 클수록 작은 객체를 탐지**
- **convolution 기반 예측** - 각 feature map에 작은 **3×3 conv filter**를 적용해 category score와 bounding box offset 예측. 별도 FC Layer가 없어 빠름
- **Default box와 종횡비** - 각 feature map 셀마다 여러 default box를 설정해 다양한 크기·종횡비의 객체 탐지

**NMS의 동작 과정**
- 모든 박스를 confidence score 기준으로 정렬하고 가장 높은 박스 선택 → 예: A(0.9), B(0.85), C(0.6) → **A 선택**
- 선택한 박스와 다른 박스들의 IoU를 계산해 임계값 이상인 것 제거
  - A-B IoU 0.7 (기준 0.5 초과) → **B 제거**
  - A-C IoU 0.3 (기준 0.5 미만) → **C 유지**
- 남은 박스 중 최고 confidence 선택 → **C(0.6)** → 반복
- 남은 박스가 없으면 종료

### 2.2 Training

**① Matching Strategy**
- 각 default box를 실제 객체의 bounding box와 매칭
- IoU를 계산해 **가장 높은 IoU를 가지는 default box**를 매칭
- IoU **0.5 이상이면 추가로 매칭**해 학습을 유연하게

**② Loss Function**
- **Localization Loss** - 실제 박스와 예측 박스의 차이를 **Smooth L1**으로. L2보다 튀는 값을 잘 반영. 값을 예측하므로 회귀 loss
- **Confidence Loss** - 예측 클래스 확률과 실제 클래스의 차이를 **Softmax Loss**로

![그림](/img/ssd/05.png)

`N`은 매칭된 default box의 개수다.

**③ Hard Negative Mining**
- IoU ≥ 0.5 → positive(객체 존재), IoU < 0.5 → negative(배경)
- positive 박스에만 localization + confidence loss 적용. negative는 classification loss만
- positive와 negative의 비율이 크게 불균형 → **negative 중 손실이 큰 상위 3배수만** 학습에 사용

**④ Data Augmentation**
- random crop과 여러 비율의 사전 변형을 적용해 다양한 크기의 객체에 대응
- 실험 결과 **mAP가 8.8% 향상**

---

## 3. Experimental Results

VOC2007 test 기준이다.

| 모델 | mAP | FPS |
|---|---|---|
| **SSD300** | 74.3% | **59** |
| **SSD512** | 76.9% | - |
| Faster R-CNN | 73.2% | 7 |
| YOLO | 63.4% | 45 |

- Nvidia Titan X 기준. **SSD300이 Faster R-CNN보다 정확하면서 8배 이상 빠르고, YOLO보다 훨씬 정확하면서 더 빠름**
- 더 큰 데이터셋으로 학습하면 SSD300 77.2%, SSD512 79.8%까지 오름

**3.2 Model analysis - 무엇이 약점인가**

![Fig. 4](/img/ssd/f4.png)

- 작은 객체에서 약함 → 큰 feature map에서만 잡히는데 그 층은 의미 정보가 얕음
- 이것이 3.6절의 augmentation 설계로 이어짐

**속도 향상의 출처**
- 근본적인 개선은 **proposal 생성과 픽셀 또는 feature resampling 단계를 없앤 것**
- 그 자리를 작은 conv filter로 대체
- 그 필터를 **여러 스케일의 feature map에 적용하고 종횡비별로 예측을 분리**한 것이 정확도를 지켜줌

---

## 정리

SSD가 보여준 것은 "**단계를 없애고 그 역할을 구조로 흡수할 수 있다**"는 것이다.

proposal 단계는 "어디를 볼지 정하는" 역할이었다. SSD는 그걸 **미리 깔아둔 default box와 다중 스케일 feature map**으로 대체한다. 어디를 볼지를 매번 계산하는 대신, 가능한 모든 곳에 후보를 미리 배치하고 한 번에 판정한다.

같은 발상은 이후 one-stage 검출기 전반의 기본 골격이 됐다.

---

*`f`·`t` 로 시작하는 그림은 원 논문에서 가져왔다. Liu et al., [SSD](https://arxiv.org/abs/1512.02325), ECCV 2016.*
