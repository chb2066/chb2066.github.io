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

> 📄 [**Emerging Properties in Self-Supervised Vision Transformers**](https://arxiv.org/abs/2104.14294) · ICCV 2021 · Caron, Touvron, Misra et al.

## Abstract

**문제**
ViT는 CNN 대비 뚜렷한 이점이 보이지 않았음. **지도학습의 라벨 신호가 ViT의 잠재력을 제한하고 있는 것 아닌가**가 이 논문의 가설

**해결책**
라벨 없는 **self-distillation** — teacher와 student에 서로 다른 크기의 crop을 주고, student가 teacher 분포를 따라가게 함
centering과 sharpening 두 장치만으로 collapse를 막음

---

## 1. Introduction

![Figure 1](/img/dinov1/f1.png)

**문제** — CNN 대비 ViT 모델의 뚜렷한 이점이 보이지 않았다.

**가설** — supervised learning을 써서 그렇다. 라벨이 주는 신호가 ViT의 잠재력을 제한하고 있다.

**해결책** — BERT나 GPT처럼 라벨 없이 이미지 자체의 구조를 학습하도록 SSL을 써보자.

**제안된 아이디어**
- **Knowledge Distillation 구조**를 씀
- **Collapse 방지** — 라벨이 있으면 역전파로 가중치가 한쪽에 쏠리지 않게 정렬되지만, 라벨이 없으면 한쪽으로만 쏠릴 수 있음

---

## 3. Approach

### 3.1 SSL with Knowledge Distillation

![Figure 2](/img/dinov1/f2.png)

**입력 요소** — 원본 이미지에서 여러 crop을 만든다.
- **Global crop 2개** — 이미지의 큰 영역. 보통 50% 이상
- **Local crop 여러 개** — 보통 6~8개. 작은 영역이고 50% 미만

**teacher는 global crop 2개만 보고, student는 global crop 2개와 local crop 전부를 본다.**

이 비대칭이 핵심이다. student가 **일부분만 보고 전체 문맥을 맞히도록** 학습되므로, 작은 영역만 보고도 전역적인 의미를 유추하는 능력이 생긴다.

- teacher는 gradient를 전달받지 않음
- student는 gradient로 학습 → `Loss = -p₂ log p₁`
- teacher는 student의 파라미터를 **EMA로 전달받아** 갱신

**특이점** — 둘 다 주어진 이미지로 원본 이미지를 맞히는 것이 목표다. 그래서 student는 작은 이미지를 보고 큰 이미지를 맞히게 되는데, 이는 사실상 **segmentation과 비슷한 학습**을 하게 만든다.

**Loss에 대한 이해**
- 정답값에 해당하는 클래스의 확률이 가장 크게 나옴
- `H` 는 전체 클래스에 대해 각 정답값과 `log Ps`(예측값)를 곱해 모두 더한 것
- **정답을 맞히는 예측값이 커질수록 전체 값이 작아짐** → 역전파에서 loss를 줄이는 방향이 곧 예측값을 키우는 방향

![그림](/img/dinov1/01.png)

**MoCo의 Momentum 차용**
$$
θ_t ← λθ_t + (1 − λ)θ_s
$$

**기존 방식과의 차이**
- queue 같은 **memory bank를 쓰지 않음**
- **contrastive loss를 쓰지 않음**

MoCo에서 "느리게 변하는 teacher"라는 장치만 가져오고 나머지는 버린 셈이다.

### Collapse와 그 해결

**문제** — loss를 낮추는 방향으로만 학습이 진행되는데, 라벨이 없으므로 **모든 값을 0으로 만들거나 항상 같은 값을 출력하는 것**도 loss를 낮추는 유효한 해가 된다.

**해결책 — centering과 sharpening.** momentum teacher의 출력에 두 연산만 적용하는데, **이 둘은 서로 반대 방향으로 작용한다.**

| | 하는 일 | 막는 붕괴 | 유도하는 붕괴 |
|---|---|---|---|
| **centering** | teacher 출력에 bias `c` 를 더함: `g(x) ← g(x) + c` | 한 차원이 지배하는 것 | **균등 분포로의 붕괴** |
| **sharpening** | teacher softmax의 temperature를 낮춤 | 균등 분포로의 붕괴 | **한 차원의 지배** |

둘을 **함께** 적용하면 두 효과가 상쇄되어 균형이 맞는다. momentum teacher가 있으면 이것만으로 collapse를 막기에 충분하다.

**centering의 성질** — center `c` 는 지수 이동 평균으로 갱신된다. **1차 배치 통계에만 의존**하므로 배치 의존이 적다. 안정성을 조금 내주고 배치 크기 의존을 줄이는 교환이다.

---

## 4. Main Results

![Table 2](/img/dinov1/t2.png)

| 평가 방식 | 성능 |
|---|---|
| k-NN 분류기 (파인튜닝 없음) | **78.3%** top-1 |
| linear evaluation (ViT-Base) | **80.1%** top-1 |

ImageNet 기준. **k-NN만으로 78.3%가 나온다는 것**이 이 논문에서 가장 자주 인용되는 결과다. 별도의 학습 없이 표현 공간의 거리만으로 분류가 된다는 뜻이고, 표현이 잘 구조화돼 있다는 직접적인 증거다.

### 4.2 Properties of ViT trained with SSL

![Figure 4](/img/dinov1/f4.png)

- **self-attention map에서 객체의 경계와 배경이 분리됨** — 라벨 학습 없이 그렇게 됨
- DINO가 학습한 특징은 분할 성능이 좋아 **k-NN만 써도 잘 됨**
- **4.2.1** 최근접 이웃 검색, **4.2.2** 장면의 의미 구조 발견, **4.2.3** downstream 전이에서 모두 확인

논문은 이 k-NN 성능이 **특정 구성 요소들이 결합됐을 때에만 창발한다**고 밝힌다. 각 장치를 따로 보면 설명되지 않는다.

---

## 5. Ablation Study of DINO

![Table 7](/img/dinov1/t7.png)

- **5.1 구성 요소의 중요도** — multi-crop과 momentum teacher가 특히 크게 기여
- **5.2 teacher 선택** — momentum teacher가 다른 선택지보다 나음
- **5.3 collapse 회피**

![Figure 7](/img/dinov1/f7.png)

teacher 목표 분포의 엔트로피와 KL이 학습 내내 어떻게 움직이는지를 보여준다. centering만 또는 sharpening만 쓰면 한쪽으로 무너지는 것이 그래프에 드러난다.

**논문이 함께 확인한 것** — predictor, 고급 정규화, contrastive loss 같은 다른 인기 있는 구성 요소들은 **안정성이나 성능 면에서 이득이 거의 없었다.** centering과 sharpening만으로 충분했다는 뜻이고, 프레임워크가 단순해진 이유이기도 하다.

- **5.5 작은 배치로 학습** — 배치 의존이 적다는 centering의 성질이 실제로 확인됨

---

## 정리

DINO에서 가져갈 것은 **반대 방향으로 작용하는 두 장치를 짝지어 균형을 잡는다**는 설계다.

centering만 쓰면 균등 분포로 무너지고, sharpening만 쓰면 한 차원이 지배한다. 각각은 실패하는데 함께 쓰면 성립한다. 붕괴를 막는 다른 방법들(contrastive loss, clustering 제약, predictor, batch norm)이 있지만, 이 조합은 **배치에 대한 의존이 가장 적다.**

그리고 부수적으로 얻은 것 — attention map에서 객체 경계가 나타난다는 관찰 — 이 오히려 더 널리 쓰이게 됐다. 라벨 없는 분할의 출발점이 여기다.

---

*`f`·`t` 로 시작하는 그림은 원 논문에서 가져왔다. Caron et al., [DINO](https://arxiv.org/abs/2104.14294), ICCV 2021.*
