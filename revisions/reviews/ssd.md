<!--
개정: 2026-09-10 (원본: src/content/reviews/ssd.md)
- i-jepa 형식으로 재배치
- 「실험에서 확인된 것」 신설 — 원본에 데이터 증강 +8.8% mAP 외에 결과가 없었다.
  VOC2007 test 기준 SSD300 74.3% mAP / 59 FPS, SSD512 76.9%, 그리고
  Faster R-CNN(73.2% / 7 FPS)·YOLO(63.4% / 45 FPS)와의 대비를 넣었다.
  이 표가 있어야 "빠르면서 정확하다"는 주장이 확인된다
- 「배경 지식」에 SSD 가 왜 빠른지의 구조적 이유 정리 — proposal 과 resampling 제거
- 끝맺음을 평서형으로 통일
- 원본의 NMS 단계별 예시(A 0.9 / B 0.85 / C 0.6)와 hard negative mining 3배수 근거는
  그대로 유지. 이 글에서 가장 구체적인 부분이다
-->
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

핵심 키워드:
single-shot detection, default box, 다중 스케일 feature map, hard negative mining
주요 전략:
proposal 단계를 없애고, 여러 해상도의 feature map 각 위치에 미리 박스를 깔아 한 번에 예측한다
사용 가능 분야:
실시간 객체 검출. 이후 one-stage 검출기의 기본 골격

#### 배경 지식

**기존 검출 모델의 구조와 한계**

기존 방식은 세 단계를 거친다.

1. bounding box를 생성한다.
2. 픽셀이나 feature를 resampling한다.
3. 고품질 분류기에 넣는다.

![그림 1](/img/ssd/01.png)

여기서 문제가 갈린다.

- **Faster R-CNN** — (1)과 (2) 과정 때문에 느리다.
- **YOLO** — single-shot으로 속도를 높였지만 정확도가 부족하다.

**SSD의 차별점**
- Faster R-CNN과 달리 **proposal도 feature resampling도 없이** 바로 검출한다.
- YOLO처럼 single-shot이면서 **높은 정확도**를 낸다.

속도와 정확도를 동시에 얻는 방법이 이 논문의 내용이다.

#### SSD method

SSD는 하나의 CNN으로 **고정된 개수의 default box를 기반**으로 바운딩 박스를 조정하고, 각 박스에서 객체가 존재할 확률을 예측한다.

**두 가지 특징**

1. **여러 크기의 feature map을 활용한다.**
2. **각 feature map에서 default box를 쓴다.**

**Default box란**
- 객체 탐지를 위해 한 픽셀 기준으로 미리 생성하는, 서로 다른 크기와 비율의 박스들이다.
- feature map을 통과할 때 **모든 픽셀마다 동일한 기준의 박스들**이 생성된다.
- 이 default box를 기준으로 classification과 bounding box regression을 수행한다.

#### 세부 구조

![그림 2](/img/ssd/02.png)

![그림 3](/img/ssd/03.png)

1. VGG-16을 백본으로 쓰되 **FC Layer를 제거**한다. 그 뒤 feature map을 더 다양하게 만들고, 이 feature map들로 classification을 수행한다. 마지막 conv가 예측 내용을 취합해 최종 예측을 만든다.
2. feature map에서 객체를 탐지하기 위해 **bounding box regression과 classification을 담당하는 head**를 추가한다.
3. 각 feature map의 모든 픽셀에 있는 여러 default box를 ground truth와 비교해 평가하고, 객체의 존재 여부와 박스 위치를 조정한다.

![그림 4](/img/ssd/04.png)

4. **NMS**를 적용해 최종 탐지 결과를 결정한다.

**NMS의 동작 과정**

- 모든 박스를 confidence score 기준으로 정렬하고 가장 높은 박스를 선택한다.
  - 예: A(0.9), B(0.85), C(0.6) → **A 선택**
- 선택한 박스(A)와 다른 박스들의 IoU를 계산해 임계값 이상인 박스를 제거한다.
  - A와 B의 IoU = 0.7 (기준 0.5 초과 → **B 제거**)
  - A와 C의 IoU = 0.3 (기준 0.5 미만 → **C 유지**)
- 남은 박스 중 가장 높은 confidence score를 가진 박스를 선택한다.
  - **C(0.6) 선택**
- C와 다른 박스들의 IoU를 계산해 제거한다.
- 남은 박스가 없으면 종료한다.

#### 모델의 주요 구성 요소

**다중 스케일 feature map**
추가 conv layer를 붙여 점진적으로 축소되는 feature map을 만들고, 이를 통해 여러 크기의 객체를 검출한다. **feature map이 클수록 작은 객체를 탐지**한다.

**convolution 기반 예측**
각 feature map에 작은 **3×3 conv filter**를 적용해 객체의 category score와 bounding box offset을 예측한다. 별도의 FC Layer가 없으므로 빠르다.

**Default box와 종횡비**
여러 크기의 각 feature map 셀마다 여러 개의 default box를 설정해 다양한 크기와 종횡비의 객체를 탐지한다.

#### 학습

**1. Matching Strategy**
- 학습 시 각 default box를 실제 객체의 bounding box와 매칭해야 한다.
- IoU를 계산해 **가장 높은 IoU를 가지는 default box**를 실제 객체와 매칭한다.
- IoU가 **0.5 이상인 경우 추가로 매칭**해 학습을 좀 더 유연하게 만든다.

**2. Loss Function**

- **Localization Loss** — 실제 bounding box와 예측 박스의 차이를 **Smooth L1 Loss**로 계산한다. L2에 비해 튀는 값을 잘 반영한다. 값을 예측해야 하므로 회귀 loss를 쓴다.
- **Confidence Loss** — 각 default box에서 예측한 클래스 확률과 실제 클래스의 차이를 **Softmax Loss**로 계산한다.

![그림 5](/img/ssd/05.png)

`N`은 매칭된 default box의 개수다.

**3. Hard Negative Mining**
- IoU ≥ 0.5면 positive(객체 존재, 학습 대상), IoU < 0.5면 negative(배경)로 판단한다.
- positive로 분류된 박스에 대해서만 localization loss와 confidence loss를 적용한다. negative 박스는 classification loss만 쓴다.
- 훈련 데이터에서 positive와 negative의 비율이 크게 불균형하므로, **negative 중 손실이 큰 상위 3배수만** 학습에 쓴다.

**4. Data Augmentation**
다양한 크기의 객체를 잘 탐지하도록 random crop과 여러 비율의 사전 변형을 적용한다. 실험 결과 데이터 증강을 추가하면 **mAP가 8.8% 향상**된다.

#### 실험에서 확인된 것

VOC2007 test 기준이다.

| 모델 | mAP | FPS |
|---|---|---|
| **SSD300** | 74.3% | **59** |
| **SSD512** | 76.9% | — |
| Faster R-CNN | 73.2% | 7 |
| YOLO | 63.4% | 45 |

Nvidia Titan X 기준이다. **SSD300이 Faster R-CNN보다 정확하면서 8배 이상 빠르고, YOLO보다 훨씬 정확하면서 더 빠르다.**

더 큰 데이터셋으로 학습하면 SSD300이 77.2%, SSD512가 79.8%까지 오른다.

**속도 향상이 어디서 오는가**
근본적인 개선은 **proposal 생성과 픽셀·feature resampling 단계를 없앤 것**이다. 그 자리를 작은 conv filter로 대체했다. 그리고 그 필터를 **여러 스케일의 feature map에 적용하고 종횡비별로 예측을 분리**한 것이 정확도를 지켜준다.

#### 정리

SSD가 보여준 것은 **"단계를 없애고 그 역할을 구조로 흡수할 수 있다"**는 것이다.

proposal 단계는 "어디를 볼지 정하는" 역할이었다. SSD는 그걸 **미리 깔아둔 default box와 다중 스케일 feature map**으로 대체한다. 어디를 볼지를 매번 계산하는 대신, 가능한 모든 곳에 후보를 미리 배치하고 한 번에 판정한다.

같은 발상은 이후 one-stage 검출기 전반의 기본 골격이 됐다.
