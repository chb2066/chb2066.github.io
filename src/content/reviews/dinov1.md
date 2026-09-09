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

**전략**
원본 이미지에서 여러 개의 crop을 만듦.
Global crop 2개 (이미지의 큰 영역, 보통 50% 이상), Local crop 여러 개 (보통 6~8개, 작은 영역, 50% 미만)
teacher: global crop 2개만 봄, student: global crop 2개 + local crop 전부를 봄
→ student가 일부분만 보고 전체 문맥을 맞추도록" 학습되기 때문에, 작은 영역만
보고도 전역적인 의미를 유추하는 능력이 생김.
teacher는 gradient 전달 안 받음.
student는 gradient를 통한 지속적 학습 진행(Loss = - p2 log p1 )
teacher는 student의 파라미터를 EMA로 전달받아 업데이트됨. student는 backprop으로 학습됨.
**특이점**
이때 둘 다 해당 이미지로 원본 이미지를 맞추는 게 p이기 때문에 student는 작은 이미지를 보고 큰 이미지로 맞추는 이미지를 맞추는 즉 seg 의 학습을 진행하게 됨.
**문제점**
collapse 발생 가능성: loss낮추는 방향의 학습이 이뤄짐. label이 없어서 모든 값을 0,0,0 혹은 동일 값으로 출력하게 되는 경우 발생 가능.
**해결책**
centering:
sharpness

---

### 개괄

문제
CNN 대비 ViT 모델의 뚜렷한 이점이 보이지 않았다.
가설
Supervised learning 써서 그렇다.
해결책
BERT나 GPT와 같은 label 없이 이미지 자체의 구조 학습을 위해 SSL 써보자.

---

제안된 아이디어
- Knowledge Distillation
- Collapse 방지(label이 었다면 역전파를 통해 가중치가 한쪽에 쏠리지 않게 정렬되지만 없다면 한쪽으로만 쏠릴 수 있음)

---

발견
ViT는 CNN과 차별되는 주요한 특징 발견
1. self attention 맵을 통해 label 학습 없이도 객체의 경계와 배경 분리 가능
  1. DINO가 학습한 특징들은 seg 분리가 좋아서 K-NN만써도 좋았다.

---

1. distillation 방법
  1. Loss에 대한 이해:
  1. 정답값에 해당하는 클래스의 확률이 가장 크게 나온다.
  1. H는 전체 클래스에 대한 각 정답값과 logPs(예측값)의 곱이다.
    1. 그래서 이걸 다 더해서 H가 출력됨
  1. 정답을 맞추는 예측 값이 커지면 커질수록 전체 값이 작아짐.
    1. 역전파할 때 loss 줄이는 방향= 예측값을 키우는 방향.

![그림 1](/img/dinov1/01.png)

1. MoCo의 Momentum 차용

$$
θt ← λθt + (1 − λ)θs
$$

  a. 기존 방식과 차이점: queue와 같은 memory bank를 사용하지 않음, contrastive loss 안씀
