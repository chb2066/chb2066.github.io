<!--
개정: 2026-09-10 (원본: src/content/reviews/dinov1.md)
- ⚠️ 원본의 「해결책」에 `centering:` `sharpness` 두 단어만 있고 설명이 비어 있었다.
  논문 3절에서 확인해 채웠다. 핵심은 두 장치가 서로 반대 방향으로 작용한다는 점이다 —
  centering 은 한 차원의 지배를 막지만 균등 분포로의 붕괴를 유도하고,
  sharpening 은 정반대다. 둘을 함께 써서 균형을 맞춘다
- 「실험에서 확인된 것」 신설 — k-NN 만으로 78.3%, linear eval 80.1% (ViT-Base).
  k-NN 성능이 좋다는 원본의 서술에 실제 수치를 붙였다
- 논문이 명시한 "다른 장치는 이득이 적다"는 관찰 추가 (predictor, 고급 정규화, contrastive loss)
- i-jepa 형식으로 재배치. 원본이 「전략」→「개괄」→「발견」 순서라 시간 흐름이 뒤집혀 있었다
- 오타 수정: "label이 었다면" → "label이 있다면"
- 끝맺음을 평서형으로 통일
- 원본의 loss 이해 서술(H가 작아지는 방향 = 예측값을 키우는 방향)은 그대로 유지
-->
---
title: DINO
paper: Emerging Properties in Self-Supervised Vision Transformers
venue: ICCV 2021
link: https://arxiv.org/abs/2104.14294
claim: 라벨 없는 self-distillation만으로도 ViT의 attention map이 객체 경계를 분리해낸다.
tags: [Self-supervised, Vision Backbone]
tier: basic
date: 2025-05-08
draft: true
---

핵심 키워드:
self-distillation, multi-crop, centering + sharpening, attention map의 창발
주요 전략:
teacher와 student가 서로 다른 크기의 crop을 보게 하고, 두 개의 반대 방향 장치로 붕괴를 막는다
사용 가능 분야:
k-NN 분류, 라벨 없는 객체 분할, 범용 백본

#### 배경 지식

**문제**
CNN 대비 ViT 모델의 뚜렷한 이점이 보이지 않았다.

**가설**
supervised learning을 써서 그렇다. 라벨이 주는 신호가 ViT의 잠재력을 제한하고 있다.

**해결책**
BERT나 GPT처럼 라벨 없이 이미지 자체의 구조를 학습하도록 SSL을 써보자.

**제안된 아이디어**
- Knowledge Distillation 구조를 쓴다.
- Collapse를 방지한다. 라벨이 있다면 역전파를 통해 가중치가 한쪽에 쏠리지 않게 정렬되지만, 라벨이 없으면 한쪽으로만 쏠릴 수 있다.

#### DINO method

**전략**

원본 이미지에서 여러 개의 crop을 만든다.

- **Global crop 2개** — 이미지의 큰 영역. 보통 50% 이상이다.
- **Local crop 여러 개** — 보통 6~8개. 작은 영역이고 50% 미만이다.

**teacher는 global crop 2개만 보고, student는 global crop 2개와 local crop 전부를 본다.**

이 비대칭이 핵심이다. student가 **일부분만 보고 전체 문맥을 맞히도록** 학습되므로, 작은 영역만 보고도 전역적인 의미를 유추하는 능력이 생긴다.

- teacher는 gradient를 전달받지 않는다.
- student는 gradient로 지속적으로 학습한다. `Loss = -p₂ log p₁`
- teacher는 student의 파라미터를 **EMA로 전달받아** 갱신된다.

**특이점**
둘 다 주어진 이미지로 원본 이미지를 맞히는 것이 목표다. 그래서 student는 작은 이미지를 보고 큰 이미지를 맞히게 되는데, 이는 사실상 **segmentation과 비슷한 학습**을 하게 만든다.

#### Collapse와 그 해결

**문제**
loss를 낮추는 방향으로만 학습이 진행되는데, 라벨이 없으므로 **모든 값을 0으로 만들거나 항상 같은 값을 출력하는 것**도 loss를 낮추는 유효한 해가 된다. 이걸 collapse라고 한다.

**해결책 — centering과 sharpening**

DINO는 momentum teacher의 출력에 두 가지 연산만 적용해서 collapse를 막는다. 그리고 **이 둘은 서로 반대 방향으로 작용한다.**

| | 하는 일 | 막는 붕괴 | 유도하는 붕괴 |
|---|---|---|---|
| **centering** | teacher 출력에 bias `c`를 더한다: `g(x) ← g(x) + c` | 한 차원이 지배하는 것 | **균등 분포로의 붕괴** |
| **sharpening** | teacher softmax의 temperature를 낮춘다 | 균등 분포로의 붕괴 | **한 차원의 지배** |

둘을 **함께** 적용하면 두 효과가 상쇄되어 균형이 맞는다. momentum teacher가 있는 상황에서는 이것만으로 collapse를 막기에 충분하다.

**centering의 성질**
center `c`는 지수 이동 평균으로 갱신된다. **1차 배치 통계에만 의존**하므로 배치에 대한 의존이 적다. 안정성을 조금 내주고 배치 크기 의존을 줄이는 교환이다.

**논문이 함께 확인한 것**
predictor, 고급 정규화, contrastive loss 같은 다른 인기 있는 구성 요소들은 **안정성이나 성능 면에서 이득이 거의 없었다.** centering과 sharpening만으로 충분했다는 뜻이고, 프레임워크가 단순해진 이유이기도 하다.

#### Distillation의 세부

**Loss에 대한 이해**

- 정답값에 해당하는 클래스의 확률이 가장 크게 나온다.
- `H`는 전체 클래스에 대해 각 정답값과 `log Ps`(예측값)를 곱한 것이다. 이것을 모두 더해 `H`가 출력된다.
- **정답을 맞히는 예측값이 커질수록 전체 값이 작아진다.** 따라서 역전파에서 loss를 줄이는 방향이 곧 예측값을 키우는 방향이다.

![그림 1](/img/dinov1/01.png)

**MoCo의 Momentum 차용**

$$
θ_t ← λθ_t + (1 − λ)θ_s
$$

**기존 방식과의 차이**
- queue 같은 memory bank를 쓰지 않는다.
- contrastive loss를 쓰지 않는다.

MoCo에서 "느리게 변하는 teacher"라는 장치만 가져오고 나머지는 버린 셈이다.

#### 실험에서 확인된 것

**ViT가 CNN과 차별되는 특징이 창발한다.**

1. **self-attention map에서 객체의 경계와 배경이 분리된다.** 라벨 학습 없이 그렇게 된다.
2. DINO가 학습한 특징은 분할 성능이 좋아서 **k-NN만 써도 잘 된다.**

**수치**

| 평가 방식 | 성능 |
|---|---|
| k-NN 분류기 (파인튜닝 없음) | **78.3%** top-1 |
| linear evaluation (ViT-Base) | **80.1%** top-1 |

ImageNet 기준이다. **k-NN만으로 78.3%가 나온다는 것**이 이 논문에서 가장 자주 인용되는 결과다. 별도의 학습 없이 표현 공간의 거리만으로 분류가 된다는 뜻이고, 표현이 잘 구조화돼 있다는 직접적인 증거다.

논문은 이 k-NN 성능이 **특정 구성 요소들이 결합됐을 때에만 창발한다**고 밝힌다. 각 장치를 따로 보면 설명되지 않는다.

#### 정리

DINO에서 가져갈 것은 **반대 방향으로 작용하는 두 장치를 짝지어 균형을 잡는다**는 설계다.

centering만 쓰면 균등 분포로 무너지고, sharpening만 쓰면 한 차원이 지배한다. 각각은 실패하는데 함께 쓰면 성립한다. 붕괴를 막는 다른 방법들(contrastive loss, clustering 제약, predictor, batch norm)이 있지만, 이 조합은 **배치에 대한 의존이 가장 적다.**

그리고 부수적으로 얻은 것 — attention map에서 객체 경계가 나타난다는 관찰 — 이 오히려 더 널리 쓰이게 됐다. 라벨 없는 분할의 출발점이 여기다.
