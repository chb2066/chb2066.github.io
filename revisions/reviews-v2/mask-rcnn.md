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

> 📄 [**Mask R-CNN**](https://arxiv.org/abs/1703.06870) · ICCV 2017 · He, Gkioxari, Dollár, Girshick

## Abstract

**문제**
instance segmentation은 검출과 분할을 동시에 해야 함. 기존 semantic segmentation은 같은 클래스의 개별 객체를 구분하지 못함
Faster R-CNN의 RoIPool은 좌표를 정수로 내림 → 검출에서는 무해했지만 **픽셀 단위 마스크 예측에서는 치명적**

**해결책**
Faster R-CNN에 마스크 예측 branch를 **병렬로** 붙이고, RoIPool의 양자화를 **RoIAlign**으로 제거
클래스 간 경쟁 없이 독립적인 binary mask를 예측

---

## 1. Introduction

![Figure 1](/img/mask-rcnn/f1.png)

**Mask R-CNN이 확장한 것**
- Faster R-CNN(anchor box 기반 two-stage 검출기)의 classification·bounding box regression에 더해, **픽셀 단위 존재 여부를 예측하는 branch를 병렬로 추가**

**기존 semantic segmentation과의 차이**
- **기존** — segmentation을 먼저 하고 classification. 존재 여부를 먼저 예측하고 클래스를 구분 → **같은 클래스의 개별 객체를 구분할 수 없음**
- **Mask R-CNN** — 클래스와 bounding box를 먼저 예측하고, 그 박스 안에서 픽셀별 존재 여부를 예측. 객체를 먼저 구분(염소 1, 염소 2) → **instance segmentation**

![그림](/img/mask-rcnn/02.png)

**본 논문의 기여**
1. 마스크 branch를 병렬로 붙여 아주 작은 오버헤드만으로 instance segmentation 달성
1. RoIAlign으로 정렬 문제를 고쳐 마스크 정확도를 크게 개선
1. 같은 구조가 사람 자세 추정에도 그대로 적용됨

---

## 2. Related Work

**RoI Pooling과 그 문제** — FC Layer를 쓰기 위해 크기를 고정하는 과정

1. 후보 영역 좌표를 feature map의 정수 좌표로 변환 → `[x/s]` 로 stride로 나누고 내림. 예: `[4/2.5] = [1.6] = 1`
1. 후보 영역을 일정 개수의 셀(spatial bin)로 나눔 → 이때도 가장 가까운 정수로 변환
1. 각 셀에서 집계(max pooling 등)해 동일 크기의 feature map으로

→ **두 번의 내림에서 공간 정보가 어긋남.** 검출에서는 몇 픽셀 오차가 문제되지 않지만 픽셀 단위 마스크에서는 치명적

---

## 3. Mask R-CNN

**Faster R-CNN 복습**
- RPN으로 bounding box 후보를 만들고, RoIPool로 각 proposal의 feature를 추출한 뒤, 객체 분류와 box regression 수행
- 백본이 만든 feature map을 RPN과 RoI Head가 **함께 사용**해 연산 최적화

**Mask R-CNN의 추가**
- RPN 수행 후, 두 출력(classification, box offset)과 **병렬로** 각 RoI에 대한 픽셀 존재 여부를 출력

**Mask Prediction Head**
- 각 RoI에 대해 `K × m²` 차원 출력 → K개 클래스마다 하나씩 `m × m` binary mask

**클래스 간 경쟁의 부재**
- K개 클래스 각각에 **독립적으로 sigmoid**를 적용해 binary mask 예측
- classification branch에서 정해진 클래스의 마스크만 최종 선택
- 논문은 이 분리가 **필수적**(essential)이라고 명시 → 클래스별로 경쟁시키면(softmax) 성능이 떨어짐

**Lmask** — 최종 선택된 클래스의 마스크에만 픽셀 단위 sigmoid를 적용하고 **binary cross-entropy**로 정의

**전체 손실** — classification + box regression + mask 세 항의 합

![그림](/img/mask-rcnn/03.png)

**FCN을 쓰는 이유**
- 각 RoI에서 `m × m` 마스크를 예측할 때 FC Layer의 flatten이 없으므로 **공간 구조가 유지됨**
- FCN 방식이 FC 방식보다 **더 적은 파라미터로 더 높은 정확도**

![Figure 4](/img/mask-rcnn/f4.png)

### RoIAlign

![Figure 3](/img/mask-rcnn/f3.png)

**필요한 이유** — pixel-to-pixel 마스크 예측에서는 공간 정렬이 정확히 유지돼야 하는데, RoIPool은 정수 변환을 거침

**핵심 아이디어**
1. **RoI 경계와 bin의 양자화 제거** → `[x/16]` 대신 `x/16` 을 그대로
1. 각 bin에서 **네 개의 정규화된 샘플링 위치** 값을 계산 → 1/4, 3/4 지점에서 샘플링
1. 샘플링 위치가 픽셀 중심이 아니면 값을 직접 읽을 수 없음 → **양선형 보간**으로 인접 네 픽셀을 가중합
   ```text
   f(xs, ys) = w11·V(x1,y1) + w21·V(x2,y1) + w12·V(x1,y2) + w22·V(x2,y2)
   ```
1. 최종적으로 max pooling으로 집계

**효과**
- 마스크 정확도를 **상대적으로 10~50% 개선**
- **localization 기준이 엄격할수록(AP75) 향상 폭이 커짐** → 정렬 문제를 고친 것이 맞다는 증거

### 3.1 Implementation Details

**백본: ResNet + FPN**
- 기존 CNN은 깊은 층 feature만 써서 해상도가 낮음 → **작은 객체를 놓침**

**FPN의 핵심 아이디어**
- **top-down path** — 상위 계층을 upsampling해 해상도를 높임
- **lateral connection** — upsampling된 특징과 하위 계층을 결합해 위치 정보를 살림

1. ResNet 각 stage에서 feature map `C2`, `C3`, `C4`, `C5` 추출
1. `C5` 부터 아래로 내려오며 결합
   - `C5` 에 1×1 conv → `P5`
   - `C4` 에 1×1 conv 후 upsampling된 `P5` 를 더함 → `P4`
   - `C3` 에 1×1 conv 후 upsampling된 `P4` 를 더함 → `P3`
1. 여러 해상도의 `P2`~`P5` 획득 → 다양한 크기의 객체를 각기 적합한 층에서 처리

![그림](/img/mask-rcnn/07.png)

**Network Head**
- RoIAlign 이후 classification과 mask 예측이 동시에 진행
- **Mask Branch** — 각 클래스별 독립적인 마스크 예측
- **Classification Branch** — 객체 클래스 확정 → 그 클래스에 맞는 마스크를 최종 선택

---

## 4. Experiments: Instance Segmentation

COCO test-dev 기준이다.

| 백본 | mask AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| ResNet-101-FPN | 35.7 | 58.0 | 37.8 | 15.5 | 38.1 | 52.4 |
| ResNeXt-101-FPN | **37.1** | 60.0 | 39.4 | 16.9 | 39.9 | 53.5 |

![Figure 6](/img/mask-rcnn/f6.png)

- 겹치는 객체에서 FCIS 계열이 보이던 체계적 artifact가 Mask R-CNN에는 없음
- **Faster R-CNN에 아주 작은 오버헤드만 더하고 5 fps로 동작** → 마스크 branch를 병렬로 붙였을 뿐이라 비용이 거의 늘지 않는 것이 설계의 강점

---

## 5. Mask R-CNN for Human Pose Estimation

- 키포인트를 **one-hot 마스크**로 보면 같은 구조가 그대로 적용됨
- 구조를 바꾸지 않고 출력 해석만 바꿔 다른 태스크로 옮긴 사례

---

## 정리

이 논문에서 가져갈 것은 두 가지다.

**첫째, 양자화를 제거하니 문제가 풀렸다는 것.** RoIPool의 내림 연산은 검출에서는 무해했지만 마스크에서는 치명적이었다. **어떤 태스크로 옮길 때 기존 부품의 근사가 여전히 무해한지 다시 물어야 한다**는 사례다.

**둘째, 병렬로 붙이고 경쟁시키지 않은 것.** 마스크와 클래스를 한 softmax로 묶지 않고 분리했다. 여러 예측을 한 목적함수에 넣을 때 **경쟁시켜야 하는지 독립시켜야 하는지**는 자명하지 않고, 여기서는 독립이 정답이었다.

---

*`f`·`t` 로 시작하는 그림은 원 논문에서 가져왔다. He et al., [Mask R-CNN](https://arxiv.org/abs/1703.06870), ICCV 2017.*
