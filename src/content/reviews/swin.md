---
title: Swin Transformer
paper: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
venue: ICCV 2021
link: https://arxiv.org/abs/2103.14030
claim: self-attention을 로컬 윈도우로 제한하고 층마다 윈도우를 shift하면, 연산량이 이미지 크기에 선형이면서도 윈도우 간 정보 교환이 유지된다.
tags: [Backbone, Transformer]
tier: main
date: 2025-03-20
draft: false
---

[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

## Abstract

**문제**
언어와 달리 이미지는 규모 변화가 크고 해상도가 높음. ViT는 토큰 수의 제곱에 비례해 연산량이 늘어 고해상도 dense prediction에 쓸 수 없음

**해결책**
self-attention을 겹치지 않는 로컬 윈도우 안으로 제한 → 이미지 크기에 선형
층마다 윈도우를 이동(shift) 시켜 윈도우 간 연결을 확보
패치를 합쳐가며 계층적 feature map 구성 → 검출·분할에 바로 붙음

---

## 1. Introduction

**이 논문에서 내가 가장 중요하게 본 것**

1. ViT는 연산량이 커서 hidden node 수를 키우는 데 부담이 있었는데, Swin은 연산량이 작아서 키울 수 있다. 같은 연산량 기준으로 한 픽셀의 정보를 더 많은 노드에 저장할 수 있으므로 픽셀 정보를 더 정확하게 뽑는다
1. 기존에는 16×16 패치를 써서 한 패치가 받을 수 있는 정보가 제한적이었다. 4×4, 8×8 같은 작은 패치를 쓰면 각 패치가 더 많고 다양한 정보를 얻어 세밀한 특징이 나온다

**해상도 문제**
- 텍스트는 토큰 수가 적지만 이미지는 해상도가 큼 → 이미지 크기에 따라 연산량이 급격히 증가
- semantic segmentation처럼 고해상도가 필요한 태스크에서 특히 문제
- 기존 ViT는 토큰 수의 제곱에 비례 → 해상도가 올라가면 감당이 안 됨

**복잡도 비교** - `h × w` 패치, 채널 `C`, 윈도우 `M × M` 일 때

```text
Ω(MSA)   = 4hwC² + 2(hw)²C      ← (hw)² 항이 문제
Ω(W-MSA) = 4hwC² + 2M²hwC       ← M 이 고정이면 hw 에 선형
```

- 윈도우 크기 `M` 은 고정(논문은 7) → 두 번째 식은 이미지 크기 `hw` 에 선형
- 윈도우 수가 `hw / M²` 이고 각 윈도우 안의 비용이 `M²` 에 비례하기 때문

---

## 2. Related Work

- conv layer를 self-attention으로 대체한 시도 - 로컬 윈도우에서 self-attention을 수행해 성능은 올랐지만 지연 문제가 생김 → Swin은 shifted window로 해결
- CNN에 self-attention이나 Transformer를 추가 - CNN 백본이나 헤드에 결합해 장거리 의존성과 이질적 상호작용을 인코딩
- **ViT가 가장 가까움** - 패치를 겹치지 않게 나눠 토큰으로 쓰고 Transformer를 그대로 분류에 적용. 다만 고해상도 입력과 세밀한 특징에 한계

---

## 3. Method

### 3.1 Overall Architecture

**계층적 feature map** - ViT와의 차이는 위로 갈수록 해상도가 감소하고 채널이 늘어난다는 점. CNN 백본의 구조를 Transformer로 되살린 것이고, 덕분에 검출·분할처럼 다중 스케일 feature가 필요한 태스크에 바로 붙는다.

**입력 요소와 흐름**
1. **토큰 처리** - Stage 1에서 입력 RGB 이미지를 겹치지 않는 작은 패치로 나눔. 각 패치를 flatten하고 임베딩 층을 거쳐 임의 차원으로 변환해 하나의 토큰으로 다룸. 패치 수는 유지
1. **Patch Merging Layer** - 깊어지면 토큰 수를 줄이며 계층적 표현을 만듦. 인접한 2×2 패치의 feature를 concat한 뒤 선형 레이어를 거치면 토큰 수 4배 감소, 채널 2배(2C) 증가. CNN에서 해상도를 낮출 때 채널을 늘리는 것과 같은 원리
1. 이 다운샘플링과 Swin 블록이 Stage 2 → 3 → 4로 이어짐

![그림](/img/swin/06.png)

**Swin Transformer 블록** - 기존 Transformer의 MSA를 Window MSA와 Shifted Window MSA로 교체한 구조

- MLP: `Linear(C, 4C) → GELU → Linear(4C, C)`
- LN: Layer Normalization. 하나의 입력 벡터(토큰)에 대한 정규화

### 3.2 Shifted Window based Self-Attention

**윈도우 안에서의 Self-Attention**
- 이미지를 겹치지 않는 고정 크기 윈도우(7×7)로 나누고 각 윈도우 안에서만 self-attention
- 224×224 기준으로 64개 윈도우가 병렬 처리됨

**윈도우 간 연결**
- 로컬 윈도우만 쓰면 윈도우 사이에 정보 교환이 안 됨
- 그래서 self-attention 층 사이에 윈도우 배치를 조금씩 이동(shift) → 토큰 전체를 옮기는 것

**cyclic shift와 masking** - 윈도우를 밀면 경계에서 크기가 안 맞는 조각이 생긴다.

- **padding** - 부족한 부분을 채움. 구현은 단순하지만 윈도우 수가 늘어남
- **cyclic shift** - 좌상단으로 순환 이동. 윈도우 수가 그대로 유지됨 ← 논문의 선택

- 다만 원래 인접하지 않던 영역이 한 윈도우에 들어옴 → 다른 윈도우의 패치끼리 attention하지 않도록 masking 적용. A끼리, B끼리, C끼리만 계산
- 이후 reverse cyclic shift로 위치 왜곡 복원
- 실측에서도 cyclic 방식이 padding보다 빠름

**상대위치 bias**
- self-attention 계산 시 상대위치 임베딩 사용
- 이미지에서는 상대적 위치에 따라 score가 달라져야 하기 때문

![그림](/img/swin/09.png)

### 3.3 Architecture Variants

| 모델 | C | 층 수 |
|---|---|---|
| Swin-T | 96 | {2, 2, 6, 2} |
| Swin-S | 96 | {2, 2, 18, 2} |
| Swin-B | 128 | {2, 2, 18, 2} |
| Swin-L | 192 | {2, 2, 18, 2} |

---

## 4. Experiments

- **4.1 ImageNet-1K 분류** - top-1 87.3%
- **4.2 COCO 검출** - 58.7 box AP, 51.1 mask AP (test-dev)
- **4.3 ADE20K 분할** - 53.5 mIoU (val)

당시 최고 성능 대비 COCO +2.7 box AP, +2.6 mask AP, ADE20K +3.2 mIoU.

분류에서의 우위보다 dense prediction에서의 격차가 훨씬 크다는 게 중요하다. 계층적 구조와 고해상도 처리 능력이 실제로 그 태스크에서 값을 한다는 뜻이다.
