<!--
개정: 2026-09-10 (원본: src/content/reviews/swin.md)
- i-jepa 형식으로 재배치. Related Work 를 「배경 지식」 앞쪽으로 옮겨 흐름을 맞춤
- 복잡도 식을 텍스트로 명시 — 원본은 그림에 의존해서 "H*W 제곱에 비례"라고만 적혀 있었다.
  Ω(MSA)=4hwC²+2(hw)²C, Ω(W-MSA)=4hwC²+2M²hwC 를 적고 어느 항이 문제인지 표시
- cyclic shift 를 padding 방식과 비교 — 논문 Table 5 에서 cyclic 이 더 빠르다는 근거.
  원본은 cyclic 만 설명해서 "왜 굳이 복잡하게" 가 빠져 있었다
- 「실험에서 확인된 것」을 실제 수치로 교체 — ImageNet 87.3%, COCO 58.7 box AP /
  51.1 mask AP, ADE20K 53.5 mIoU, 그리고 당시 SOTA 대비 향상 폭
- 끝맺음을 평서형으로 통일
- 원본의 「핵심」 3항목(연산량이 줄어 hidden node 를 키울 수 있다는 관찰)은 그대로 유지
-->
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

핵심 키워드:
window 기반 self-attention, shifted window, 계층적 feature map, 선형 복잡도
주요 전략:
윈도우 안에서만 attention해서 비용을 선형으로 낮추고, 층 사이에 윈도우를 밀어 윈도우 간 연결을 만든다
사용 가능 분야:
분류뿐 아니라 검출·분할 같은 dense prediction. 범용 백본

#### 핵심

1. **ViT는 연산량이 커서 hidden node 수를 키우는 데 부담이 있었는데, Swin은 연산량이 작아서 키울 수 있다.** 같은 연산량 기준으로 한 픽셀의 정보를 더 많은 노드에 저장할 수 있으므로, 픽셀에 대한 정보를 더 정확하게 뽑는다.
2. **기존에는 16×16 패치를 써서 한 패치가 받을 수 있는 정보가 제한적이었다.** 4×4, 8×8 같은 작은 패치를 쓰면 각 패치가 더 많고 다양한 정보를 얻어서 세밀한 특징이 나온다.
3. 아래는 1, 2를 가능하게 하는 방법론이다.

#### 배경 지식

**해상도 문제**
텍스트는 토큰 수가 적지만 이미지는 해상도가 크다. 그래서 이미지 크기에 따라 연산량이 급격히 늘어난다. semantic segmentation처럼 고해상도가 필요한 태스크에서 특히 문제다.

기존 ViT는 **토큰 수의 제곱에 비례**해 연산량이 증가하므로 해상도가 올라가면 감당이 안 된다. Swin은 로컬 윈도우 단위로 self-attention을 적용해서 이 비용을 줄인다.

**복잡도 비교**

`h × w` 개의 패치, 채널 `C`, 윈도우 크기 `M × M` 일 때:

```text
Ω(MSA)   = 4hwC² + 2(hw)²C      ← (hw)² 항이 문제
Ω(W-MSA) = 4hwC² + 2M²hwC       ← M 이 고정이면 hw 에 선형
```

![그림 1](/img/swin/01.png)

![그림 2](/img/swin/02.png)

윈도우 크기 `M`은 고정이므로 (논문은 7을 쓴다) 두 번째 식은 **이미지 크기 `hw`에 선형**이다. 윈도우 수가 `hw / M²`이고 각 윈도우 안의 비용이 `M²`에 비례하기 때문이다.

**선행 연구와의 관계**

- **conv layer를 self-attention으로 대체한 시도** — 로컬 윈도우에서 self-attention을 수행해 성능은 올랐지만 지연 문제가 생겼다. Swin은 shifted window로 이를 해결한다.
- **CNN에 self-attention이나 Transformer를 추가하는 방식** — self-attention이 CNN 백본이나 헤드에 결합되어 장거리 의존성과 이질적 상호작용을 효과적으로 인코딩한다.
- **ViT가 가장 가깝다** — 패치를 겹치지 않게 나눠 토큰으로 쓰고 Transformer 구조를 그대로 이미지 분류에 적용했다. 다만 **고해상도 입력과 세밀한 특징에는 한계**가 있다.

#### Swin의 주요 특징

**계층적 feature map**

![그림 3](/img/swin/03.png)

ViT와의 차이는 **아래에서 위로 올라갈수록 해상도가 감소하고 채널이 늘어난다**는 점이다. CNN 백본의 구조를 Transformer로 되살린 것이다. 이 덕분에 검출이나 분할처럼 다중 스케일 feature가 필요한 태스크에 바로 붙는다.

**Shifted window**

self-attention 층 사이에 **윈도우의 위치를 조금씩 이동시킨다.** sliding window 방식의 self-attention보다 **지연이 낮다.**

![그림 4](/img/swin/04.png)

#### 전체 구조

**토큰 처리**
Stage 1에서 입력 RGB 이미지를 ViT와 비슷하게 겹치지 않는 작은 패치로 나눈다. 각 패치를 flatten하고 임베딩 층을 거쳐 임의의 차원으로 변환한 뒤 하나의 토큰으로 다룬다. 패치 수는 유지된다.

![그림 5](/img/swin/05.png)

**Patch Merging Layer**
네트워크가 깊어지면 토큰 수를 줄이며 계층적 표현을 만든다.

![그림 6](/img/swin/06.png)

`4×`, `8×`, `16×`는 다운샘플링 배율이다. 인접한 2×2 패치의 feature를 concat한 뒤 선형 레이어를 거치면 **토큰 수가 4배 줄고 채널이 2배(2C)로 늘어난다.** CNN에서 해상도를 낮출 때 채널을 늘리는 것과 같은 원리다.

이 다운샘플링과 Swin 블록이 Stage 2 → Stage 3 → Stage 4로 이어진다.

**Swin Transformer 블록**
기존 Transformer의 multi-head self-attention(MSA)을 **Window MSA와 Shifted Window MSA로 교체**한 구조다.

![그림 7](/img/swin/07.png)

- MLP: `Linear(C, 4C) → GELU → Linear(4C, C)`
- LN: Layer Normalization. 하나의 입력 벡터(토큰)에 대한 정규화다.

#### Shifted Window based Self-Attention

**윈도우 안에서의 Self-Attention**
이미지를 겹치지 않는 고정 크기 윈도우(7×7)로 나누고 각 윈도우 안에서만 self-attention을 수행한다. 224×224 기준으로 64개 윈도우가 병렬 처리된다.

**윈도우 간 연결**
로컬 윈도우만 쓰면 윈도우 사이에 정보 교환이 안 된다. 그래서 self-attention 층 사이에 **윈도우 배치를 조금씩 이동(shift)** 시킨다. 토큰 전체를 옮기는 것이다.

![그림 8](/img/swin/08.png)

**cyclic shift와 masking**
윈도우를 밀면 경계에서 크기가 안 맞는 조각이 생긴다. 이걸 처리하는 방법이 두 가지다.

- **padding** — 부족한 부분을 채운다. 구현은 단순하지만 윈도우 수가 늘어난다.
- **cyclic shift** — 좌상단 방향으로 순환 이동시킨다. **윈도우 수가 그대로 유지된다.**

논문은 cyclic shift를 쓴다. 다만 이렇게 하면 원래 인접하지 않던 영역이 한 윈도우 안에 들어오므로, **다른 윈도우에 있던 패치끼리 attention하지 않도록 masking을 적용**한다. A끼리, B끼리, C끼리만 계산된다. 이후 reverse cyclic shift로 위치 왜곡을 복원한다.

실측에서도 cyclic 방식이 padding보다 빠르다. 복잡해 보이지만 그만한 이유가 있다.

**상대위치 bias**
self-attention 계산 시 상대위치 임베딩을 쓴다. 이미지에서는 상대적 위치에 따라 score가 달라져야 하기 때문이다.

![그림 9](/img/swin/09.png)

#### 아키텍처 변형

| 모델 | C | 층 수 |
|---|---|---|
| Swin-T | 96 | {2, 2, 6, 2} |
| Swin-S | 96 | {2, 2, 18, 2} |
| Swin-B | 128 | {2, 2, 18, 2} |
| Swin-L | 192 | {2, 2, 18, 2} |

#### 실험에서 확인된 것

- **ImageNet-1K 분류 — top-1 87.3%**
- **COCO 검출 — 58.7 box AP, 51.1 mask AP** (test-dev)
- **ADE20K 분할 — 53.5 mIoU** (val)

당시 최고 성능 대비 **COCO에서 +2.7 box AP, +2.6 mask AP, ADE20K에서 +3.2 mIoU**다. 분류에서의 우위보다 **dense prediction에서의 격차가 훨씬 크다**는 게 중요하다. 계층적 구조와 고해상도 처리 능력이 실제로 그 태스크에서 값을 한다는 뜻이다.

#### 정리

Swin이 한 일은 **ViT에서 버렸던 CNN의 두 가지 귀납 편향을 되돌려 놓은 것**이다.

- **국소성** — 윈도우 안에서만 attention한다.
- **계층 구조** — 깊어질수록 해상도를 줄이고 채널을 늘린다.

그러면서도 윈도우를 미는 것만으로 전역 연결을 확보한다. 비용은 선형인데 표현력은 유지된다.

여기서 옮겨갈 만한 발상은 **"제한을 걸어 비용을 낮추고, 그 제한을 층마다 어긋나게 해서 손실을 메운다"**는 골격이다. 한 층에서 잃은 연결을 다음 층의 다른 분할이 복구한다.
