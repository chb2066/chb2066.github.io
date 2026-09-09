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

#### 핵심

1. ViT는 연산량이 커서  hidden node 수를 키우는 것에 대한 부담이 있었으나 Swin은 연산량이 작아서 hidden node 수를 키울 수 있었다.(동일 연산량 기준 한 개의 픽셀정보를 더 많은 노드에 저장할 수 있어 픽셀에 대한 정보를 더 정확하게 뽑을 수 있음)
1. 기존에는 16*16 패치를 사용함으로써 한 패치가 받을 수 있는 정보가 15개의 패치에 대한 정보만 있었으나 4*4, 8*8 등의 패치를 사용함으로써 각 패치가 더 많은, 더 다양한 정보를 얻을 수 있게 되어 세밀한 특징을 얻을 수 있게 되었다.
1. 아래는 1, 2를 위한 방법론
**해상도 문제**
텍스트는 해상도가 작지만 이미지는 해상도가 크다.
→이미지 크기에 따라 연산량 급격히 증가(ex: semantic segmentation)
- 기존 ViT는 토큰 수의 제곱에 비례하여 연산량이 증가하여 해상도가 증가할 때 연산량 급격히 증가
  →Swin Transformer는 로컬 윈도우 단위로 self-attention을 적용하여 연산량을 줄임.

ViT 연산량

![그림 1](/img/swin/01.png)

H*W의 제곱에 비례(토큰 수 제곱에 비례)
Swin 연산

![그림 2](/img/swin/02.png)

H*W (이미지 크기)에 선형적(M*M은 윈도우 크기, H*W/M^2=윈도우 수)

#### Swin의 주요 특징

**hierarchical feature map**

![그림 3](/img/swin/03.png)

Swin과 ViT의 차이점:
Swin은 아래에서 위로 올라갈수록 해상도가 감소하고 채널이 늘어난다
S**hifted window**
self-attention 층 사이에 **윈도우의 위치를 조금씩 이동(shift)** 시키는 것이다
sliding window 방식의 self-attention보다 **지연(latency)이 낮다**

![그림 4](/img/swin/04.png)

## Related Work

1. conv layer를 self-attention으로 대체한 시도 존재
이 연구에서는 로컬 윈도우에서 self-attention을 수행해 성능이 향상되었지만 지연 문제가 발생함
→ Shifted window 방식으로 해결함

1. CNN에 self-attention이나 트랜스포머를 추가하는 방식
self-attention이 CNN 백본이나 헤드에 결합되어 장거리 의존성이나 이질적 상호작용을 효과적 인코딩 가능하게 해준다.

1. ViT 가 가장 연관있다.
패치를 안겹치게 나누어 토큰으로 쓰고 트랜스포머 구조를 그대로 이미지 분류 적용했다.
- 하지만 고해상도 입력, 세밀 특징에는 한계 존재

## Method

#### Overall Architecture

**토큰 처리 방식**
Stage1에서 입력 RGB이미지는 ViT와 유사하게 겹치지 않는 작은 패치들로 나뉜다.
각 패치는 flatten하고 임베딩 층을 거쳐 임의의 차원 수로 변환된 뒤, 하나의 토큰처럼 간주되어 사용한다.(패치 수는 유지된다.)

![그림 5](/img/swin/05.png)

네트워크가 깊어질수록 **Patch Merging Layer**를 통해 토큰 수를 줄이며 계층적 표현을 만든다.

![그림 6](/img/swin/06.png)

4*,8*,16* 는 다운샘플링 배율
예를 들어, 인접한 2*2 개의 패치의 피처를 concat한 후 선형 레이어를 거치면 토큰 수가 4배 줄어들고, 채널(차원 수)는 2배(2C)로 증가시킨다.
- CNN에서 해상도 낮출 때 채널 늘리는 원리
이러한 다운샘플링 및 Swin 블록을 거친 구조들이 순차적으로 **Stage 2 → Stage 3 → Stage 4**로 이어진다.

**Swin Transformer 블록**
Swin Transformer 블록은 기존 트랜스포머의 multi-head self-attention(MSA)을 Window MSA, Shifted Window MSA로 교체한 구조

![그림 7](/img/swin/07.png)

MLP: Linear(C, 4C)   →   GELU   →   Linear(4C, C)
LN = Layer Normalization(하나의 입력 벡터(토큰에 대한 정규화))

#### **Shifted Window based Self-Attention**

**Non-overlapping Window 내에서의 Self-Attention**
이미지를 겹치지 않는 고정된 크기의 윈도우로(7*7) 나누고, 각 윈도우 내에서만 self-attention을 수행(계산 복잡도 줄임, 224*224 기준 64개 병렬 수)
**윈도우 간 연결을 위한 Shifted Window**
로컬 윈도우만으로 윈도우 간 정보 교환 불가능
→self-attention 층 간에 윈도우 배치를 조금씩 이동(shift, 토큰 전체 이동) 시키는 전략을 사용

![그림 8](/img/swin/08.png)

윈도우 cyclic 이동 시 겹치지 않는 윈도우가 같은 윈도우 안에 들어옴
- 윈도우 간 정보 교환 가능
- 다른 윈도우에 있던 패치 간 attention을 막기 위해 masked를 적용해 attention한다.(A끼리, B끼리, C끼리 가능)

reverse를 통해 shift 과정에서 생긴 위치 왜곡 복원
self-attention 계산 시 상대위치 임베딩 사용(상대위치 바이어스 사용, 이미지에서는 상대적 위치에 따라 score가 달라져서 적용함.)

![그림 9](/img/swin/09.png)

#### **Architecture Variants**

1. **Swin-T**: C = 96, layer 수 = {2, 2, 6, 2}
1. **Swin-S**: C = 96, layer 수 = {2, 2, 18, 2}
1. **Swin-B**: C = 128, layer 수 = {2, 2, 18, 2}
1. **Swin-L**: C = 192, layer 수 = {2, 2, 18, 2}

#### Experiments

ImageNet, COCO object detection, ADE20K semantic segmentation등의 데이터 셋에서 우수한 성능을 보임
