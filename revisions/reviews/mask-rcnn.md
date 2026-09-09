<!--
개정: 2026-09-10 (원본: src/content/reviews/mask-rcnn.md)
- i-jepa 형식으로 재배치. Introduction / Related work / Mask R-CNN 구성을
  배경 지식 → method → 세부 아키텍처 → 실험 순으로 정리
- RoIAlign 의 효과를 수치로 보강 — 마스크 정확도를 상대적으로 10~50% 개선.
  원본은 "큰 성능 향상"이라고만 적혀 있었다 (논문 초록·1절)
- 「실험에서 확인된 것」 신설 — COCO mask AP 35.7 (ResNet-101-FPN), 37.1 (ResNeXt-101-FPN),
  5 fps. 원본에 결과가 아예 없었다
- 클래스 간 경쟁을 없앤 것이 "essential" 이라는 논문의 표현 반영. 원본의 서술이 맞다는 근거
- 끝맺음을 평서형으로 통일
- 원본의 RoIAlign 양선형 보간 전개와 FPN 단계 추적(C5→P5→P4→P3)은 그대로 유지.
  이 글에서 가장 구체적인 부분이다
-->
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

핵심 키워드:
instance segmentation, RoIAlign, 마스크 branch 병렬 추가, 클래스별 독립 sigmoid
주요 전략:
Faster R-CNN에 마스크 예측을 병렬로 붙이되, 픽셀 정렬을 깨뜨리던 정수 양자화를 제거한다
사용 가능 분야:
instance segmentation, 객체 검출, 사람 자세 추정

#### 배경 지식

**Mask R-CNN이 무엇을 확장했나**
Faster R-CNN을 확장한 모델이다. 기존의 classification과 bounding box regression에 더해, **객체의 픽셀 단위 존재 여부를 예측하는 branch를 병렬로 추가**한다. Faster R-CNN 자체는 anchor box 기반의 two-stage 검출 모델이다.

![그림 1](/img/mask-rcnn/01.png)

**기존 semantic segmentation 접근법과의 차이**

- **기존** — segmentation을 먼저 하고 classification을 한다. 객체 존재 여부를 먼저 예측하고 클래스를 구분하므로 **같은 클래스의 개별 객체를 구분할 수 없다.**
- **Mask R-CNN** — 어떤 클래스인지와 bounding box를 먼저 예측하고, 그 박스 안에서 각 픽셀의 객체 존재 여부를 예측한다. 객체를 먼저(염소 1, 염소 2처럼) 구분하므로 **instance segmentation**이 된다.

![그림 2](/img/mask-rcnn/02.png)

**RoI Pooling과 그 문제**

RoI Pooling은 FC Layer를 쓰기 위해 크기를 고정하는 과정이다.

1. 후보 영역의 좌표를 feature map의 정수 좌표로 변환한다. `[x/s]`로 stride로 나누고 내림한다. 예를 들어 `[4/2.5] = [1.6] = 1`이다.
2. 후보 영역을 일정한 개수의 셀(spatial bin)로 나눈다. 이때도 가장 가까운 정수 좌표로 변환한다.
3. 각 셀 안에서 집계(max pooling 등)해 동일한 크기의 feature map으로 만든다.

문제는 **두 번의 내림 과정에서 공간 정보가 어긋난다**는 것이다. 검출에서는 몇 픽셀의 오차가 크게 문제되지 않지만, **픽셀 단위 마스크 예측에서는 치명적**이다.

#### Mask R-CNN method

**Faster R-CNN 복습**
RPN으로 bounding box 후보를 만들고, RoIPool로 각 region proposal의 feature를 추출한 뒤, 객체 분류와 bounding box regression을 수행한다. 백본을 거쳐 생성된 feature map을 RPN과 RoI Head가 **함께 사용**해서 연산을 최적화한다.

**Mask R-CNN의 추가**
먼저 RPN을 수행한다. 이후 두 개의 출력(classification, bounding box offset)과 **병렬로** 각 RoI에 대한 객체 존재 여부를 출력한다.

**Mask Prediction Head**
mask branch는 각 RoI에 대해 `K × m²` 차원의 출력을 만든다. K개 클래스마다 하나씩, `m × m` 크기의 binary mask를 포함한다.

**클래스 간 경쟁이 없다**
mask branch는 K개 클래스 각각에 대해 **독립적으로 sigmoid를 적용해** binary mask를 예측한다. 그리고 classification branch에서 정해진 클래스에 해당하는 마스크만 최종적으로 선택한다.

논문은 이 분리가 **필수적(essential)**이라고 명시한다. 클래스별로 경쟁시키면(softmax를 쓰면) 성능이 떨어진다. mask 예측과 class 분류를 분리한 것이 성능의 상당 부분을 만든다.

**Lmask**
RoI에서 최종 선택된 클래스에 대한 마스크만 픽셀 단위 sigmoid를 적용하고 loss 계산에 쓴다. **binary cross-entropy loss**로 정의한다.

**전체 손실 함수**

![그림 3](/img/mask-rcnn/03.png)

classification, box regression, mask 세 항의 합이다.

**FCN을 쓰는 이유**
각 RoI에서 `m × m` 마스크를 예측하기 위해 FCN을 쓴다. FC Layer의 flatten 과정이 없으므로 **공간 구조가 유지**된다.

![그림 4](/img/mask-rcnn/04.png)

FCN 방식이 FC 방식보다 **더 적은 파라미터로 더 높은 정확도**를 낸다. 마스크 예측 기준이다.

#### RoIAlign

**왜 필요한가**
pixel-to-pixel 방식의 마스크 예측에서는 공간 정렬이 정확하게 유지돼야 한다. 그런데 RoIPool은 정수형 변환을 거치므로 픽셀 단위 예측에서 손실이 크다.

**핵심 아이디어**

1. **RoI 경계와 bin에 대한 양자화를 제거한다.** 좌표 `[x/16]` 대신 `x/16`을 그대로 쓴다.
2. 각 RoI bin에서 **네 개의 정규화된 샘플링 위치**의 값을 계산한다. 1/4과 3/4 지점에서 샘플링해 더 정밀하게 만든다.
3. 샘플링 위치가 픽셀 중심이 아니면 값을 직접 읽을 수 없다. 그래서 **양선형 보간(bilinear interpolation)**으로 인접한 네 픽셀 값을 가중합해 추정한다.

```text
f(xs, ys) = w11·V(x1,y1) + w21·V(x2,y1) + w12·V(x1,y2) + w22·V(x2,y2)
```

`w` 값들은 샘플링 위치와의 거리에 따라 결정되는 가중치다.

4. 최종적으로 max pooling으로 집계한다.

![그림 5](/img/mask-rcnn/05.png)

**효과**
RoIAlign은 **마스크 정확도를 상대적으로 10~50% 개선**한다. 그리고 **localization 기준이 엄격할수록(AP75 같은 지표) 향상 폭이 커진다.** 정렬 문제를 고친 것이 맞다는 증거다.

#### 네트워크 구조

**백본**
ResNet 같은 백본에 **FPN(Feature Pyramid Network)**을 추가한다.

이유는 명확하다. 기존 CNN은 깊은 층의 feature만 쓰므로 해상도가 낮아 **작은 객체를 놓친다.**

![그림 6](/img/mask-rcnn/06.png)

**FPN의 핵심 아이디어**
- **top-down path** — 상위 계층을 upsampling해 해상도를 높인다. 작은 객체 탐지에 도움이 된다.
- **lateral connection** — upsampling된 특징과 하위 계층을 결합해 위치 정보를 살린다.

단계는 이렇게 진행된다.

1. ResNet의 각 stage에서 feature map을 뽑는다. `C2`, `C3`, `C4`, `C5`.
2. 고수준 특징인 `C5`부터 아래로 내려오며 upsampling해 저수준 특징과 결합한다.
   - `C5`에 1×1 conv를 적용해 `P5` 생성
   - `C4`에 1×1 conv를 적용한 뒤 upsampling된 `P5`를 더해 `P4` 생성
   - `C3`에 1×1 conv를 적용한 뒤 upsampling된 `P4`를 더해 `P3` 생성
3. 최종적으로 여러 해상도의 feature map `P2`, `P3`, `P4`, `P5`를 얻는다.

![그림 7](/img/mask-rcnn/07.png)

다양한 크기의 객체를 각기 적합한 층에서 처리할 수 있게 된다.

**Network Head**

![그림 8](/img/mask-rcnn/08.png)

- RoIAlign 이후 classification과 mask 예측이 동시에 진행된다.
- **Mask Branch** — RoI에 대한 각 클래스별 독립적인 마스크를 예측한다.
- **Classification Branch** — 객체 클래스를 확정한다. 확정된 클래스에 맞는 마스크를 최종 선택해 출력한다.

![그림 9](/img/mask-rcnn/09.png)

#### 실험에서 확인된 것

COCO test-dev 기준 instance segmentation 결과다.

| 백본 | mask AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| ResNet-101-FPN | 35.7 | 58.0 | 37.8 | 15.5 | 38.1 | 52.4 |
| ResNeXt-101-FPN | **37.1** | 60.0 | 39.4 | 16.9 | 39.9 | 53.5 |

**Faster R-CNN에 아주 작은 오버헤드만 더하고 5 fps로 동작한다.** 마스크 branch를 병렬로 붙였을 뿐이라 비용이 거의 늘지 않는다는 점이 설계의 강점이다.

#### 정리

이 논문에서 가져갈 것은 두 가지다.

**첫째, 양자화를 제거하니 문제가 풀렸다는 것.** RoIPool의 내림 연산은 검출에서는 무해했지만 마스크에서는 치명적이었다. **어떤 태스크로 옮길 때 기존 부품의 근사가 여전히 무해한지 다시 물어야 한다**는 사례다.

**둘째, 병렬로 붙이고 경쟁시키지 않은 것.** 마스크와 클래스를 한 softmax로 묶지 않고 분리했다. 여러 예측을 한 목적함수에 넣을 때 **경쟁시켜야 하는지 독립시켜야 하는지**는 자명하지 않고, 여기서는 독립이 정답이었다.
