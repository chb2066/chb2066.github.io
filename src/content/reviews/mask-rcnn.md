---
title: Mask R-CNN
paper: Mask R-CNN
venue: ICCV 2017
link: https://arxiv.org/abs/1703.06870
claim: Faster R-CNN에 마스크 branch를 병렬로 붙이고 RoIPool의 정수 양자화를 RoIAlign으로 없애면 instance segmentation이 된다.
tags: [Segmentation, Instance Segmentation]
tier: main
date: 2025-03-12
draft: false
---

## Introduction

- Mask R-CNN은 Faster R-CNN을 확장한 모델,  기존 Classification 및 bounding box regression와 병렬로 객체 존재 여부를 예측하는 새로운 branch를 추가하였다.
  - Faster R-CNN은 anchor box 기반의 Two-stage object detection모델이다.
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

- 새로운 Branch의 Mask 예측을 위해 기존 Faster R-CNN의 RoIPool 대신 RoIAlign을 제안

ROI Pooling:
FC Layer 사용을 위해 크기를 고정하는 과정
- ROI Pooling 과정:
  1. 후보 영역의 좌표를 Feature Map의 **정수 좌표로 변환**,  [x=x/s] 좌표를 stride로 나누어 변환
    1. ex [4/2.5]=[1.6]=1
  1. Spatial bins(격자)로 후보 영역을 일정한 개수의 셀로 나눔. 영역도 가장 가까운 정수 좌표로 변환
  1. 각 셀 내에서 Aggregation(집계) 수행(Max pool 등을 사용)→동일한 크기의 Feature map으로 변환

**기존 Faster R-CNN의 RoIPool의 문제점:**
공간 정보가 유지되지 못한다.
**RoIAlign에 의한 개선점:**
정량화 없는 RoIAlign을 도입하여 공간적 정보를 유지하도록 설계되었고 Localization 기준이 엄격할수록 큰 성능 향상을 보였다.

> *[그림 자리 — Notion 원본에서 옮겨야 함]*

---

## Related work

**기존 semantic segmentation 접근법**
segmentation→Classification
→객체 존재 여부를 먼저 예측하고 class를 구분하여 동일 클래스 구분이 불가능하다.

**Mask R-CNN의 접근 방식**
어떤 class인지와 bounding box를 먼저 예측 후 박스 안 각 픽셀의 객체 존재 여부 예측
→ 객체들을 먼저(염소 1, 염소2처럼) 예측하기 때문에 instance segmentation의 효과를 갖는다.
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

---

## Mask R-CNN

#### 1. **Faster R-CNN**

RPN을 통해 객체 bounding box를 생성, RoIPool을 사용하여 각 region proposal의 feature을 추출 후 객체 분류 및 bounding box regression을 수행한다.
- 특장점: Backbone을 거쳐 생성된 Feature map을  RPN과 RoI Head가 함께 사용하여 연산 최적화

#### **2. Mask R-CNN **

Mask R-CNN은 먼저 Region Proposal Network (RPN)를 진행. 이후 두 개 출력(classification, bounding box offset prediction)과 병렬로 각 RoI(Region of Interest)에 대한 객체 존재 여부도 출력

**Mask Prediction Head:**
mask branch는 각 RoI에 대해 K*m^2차원의 출력을 생성(K개 클래스 마다 하나씩 m*m크기의 binary mask를 포함), RoI 내부에 객체의 존재 여부를 출력.
**클래스 간 경쟁 없음**
mask branch는 K개 클래스 각각에 대해 독립적으로 sigmoid를 적용해 binary mask를 예측함.  즉, classification branch에서 정해진 클래스에 해당하는 마스크만 최종적으로 선택해서 사용.
→ mask 예측과 class 분류를 분리하여 성능을 높임, 클래스별로 경쟁시킬 시 성능이 떨어짐

**Lmask(mask에 대한 loss):**
RoI에서 최종 선택된 클래스에 대한 마스크만 pixel 단위 sigmoid function를 적용하고 loss계산에 사용, Lmask를 binary cross-entropy loss로 정의
**ROI에 대한 손실 함수(Multi-Task Loss):**
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

**FCN:**
각 RoI(Region of Interest)에서 m×m 크기의 마스크를 예측하기 위해 FCN를 사용한다. 이 방식은 공간 구조 유지할 수 있도록 한다.(FC Layer의 Flatten과정이 없기 때문)
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

FCN 방식이 FC 방식보다 더 적은 parameters로 더 높은 정확도(accuracy)를 제공한다.(Mask Prediction 기준)

#### **RoIAlign**

도입:
pixel to pixel방식의 mask prediction 에서는 공간 정렬이 정확하게 유지되어야 한다.
한계:
RoIPool은 정수형 변환을 거치기 때문에 픽셀 단위 mask 예측에서는 손실이 크다.(x=[x/s])
**해결책:**
RoIAlign Layer를 제안, 강제로 정량화 하는 것이 아닌 추출된 feature과 input data 간의 정렬을 수행
**핵심 아이디어**
- RoI 경계 및 bin에 대한 정량화를 제거 (즉, 좌표 [x/16] 대신 x/16 사용)
- 각 RoI bin(구역)에서 네 개의 정규화된 샘플링 위치에서 입력 특징 값을 계산(1/4과 3/4 지점에서 샘플링해서 더 정밀한 예측 가능 )
- 이때, 양선형 보간(Bilinear Interpolation)을 사용하여 정확한 값 계산
  - 샘플링 위치가 픽셀 중심이 아니면 위치 값을 못얻는다. 따라서 인접한 네 개의 픽셀 값을 가중합하여 샘플링 위치의 값을 추정 (xs,ys)=w11V(x1,y1)+w21V(x2,y1)+w12V(x1,y2)+w22V(x2,y2) 여기서 w 값들은 샘플링 위치와의 거리에 따라 결정되는 가중치
- 최종적으로, max pooling을 적용하여 결과를 집계
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

#### **Network architecture **

Mask R-CNN 구성:
1. Backbone architecture: 전체 이미지에서 feature를 추출하는 CNN 구조
1. 네트워크 헤드(Network Head):
  - 각 RoI(Region of Interest)에 대해 바운딩 박스 인식(클래스 분류 및 회귀) 수행
  - 마스크 예측 수행

1. **Backbone architecture**
Resnet과 같은 Backbone 에 보강을 위한 FPN(**Feature Pyramid Network**) 추가하여 사용.
이유: 기존 CNN의 경우 Deep feature만 사용하기 때문에 해상도가 낮아 작은 객체 탐지가 어렵다.

> *[그림 자리 — Notion 원본에서 옮겨야 함]*

**FPN 핵심 아이디어**
top-down path
상위 계층을 upsampling하여 해상도 증가시킴(작은 객체 탐지 도움).
lateral connections
upsampling된 특징과 하위 계층을 결합하여 위치정보 활용
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

1. Resnet의 stage에서 feature map 추출, C2, C3, C4, C5 등등
1. 고수준 특징(C5)에서 부터 아래로 내려오면서 upsampling 후 저수준 저수준 특징에 결합
  1. C5에 1*1 Conv를 적용해 P5생성
  1. C4에 1*1 Conv를 적용한 뒤 업샘플링한 P5를 더해 P4 생성
  1. C3에 1*1 Conv를 적용한 뒤 업샘플링한 P4를 더해 P3 생성
최종적으로 **여러 해상도의 피처 맵 P2,P3,P4,P5를 출력**
→ 이를 통해 다양한 크기의 객체 탐지가 가능해진다.
> *[그림 자리 — Notion 원본에서 옮겨야 함]*

1. **Network Head**
- RoI Align 이후 Classification 과 mask 예측이 동시 진행.
- Mask Branch에서는 RoI(후보 영역)에 대한 각 클래스별 독립적인 Mask(객체 존재 여부)를 예측
- Classification Branch에서 객체 확정 시 해당 클래스에 맞는 마스크를 최종 선택하여 출력
