---
title: iBOT
paper: "iBOT: Image BERT Pre-Training with Online Tokenizer"
venue: ICLR 2022
link: https://arxiv.org/abs/2111.07832
claim: 별도로 사전학습한 고정 토크나이저 없이, EMA teacher를 온라인 토크나이저로 삼아 MIM과 self-distillation을 함께 최적화한다.
tags: [Self-supervised, Masked Image Modeling]
tier: main
date: 2025-05-14
draft: false
---

[iBOT: Image BERT Pre-Training with Online Tokenizer](https://arxiv.org/abs/2111.07832)

## Abstract

**문제**
BERT식 masked modeling을 이미지로 옮기려면 시각 토크나이저가 필요한데, 언어와 달리 이미지는 연속적이라 의미 단위를 뽑기 어려움
기존 방법은 픽셀을 그대로 목표로 삼거나 미리 학습한 토크나이저를 쓰는데, 둘 다 다단계 파이프라인을 강요함

**해결책**
EMA teacher를 온라인 토크나이저로 삼아 MIM과 함께 학습 → 토크나이저 사전학습 단계가 사라짐
MIM을 "토크나이저로부터의 knowledge distillation"으로 정식화

---

## 1. Introduction

**두 가지 아이디어**

**DINO의 loss**
- teacher와 student가 각각 다른 뷰를 보게 하고, student가 teacher의 확률 분포를 따라가게 만듦
- EMA로 teacher의 업데이트량을 조절
- teacher는 큰 뷰를, student는 작은 뷰를 보게 해서 부분에서 전체를 맞히도록 학습

**iBOT의 MIM**
- teacher는 마스킹하지 않은 원본 이미지(global view만)를, student는 마스킹된 이미지를 봄
- student가 보는 뷰는 global과 local을 섞는데 local이 훨씬 많음. 대략 2:8 ~ 2:10
- global을 넣는 이유는 그게 없으면 학습이 안 되기 때문 → 부분과 전체의 관계를 볼 수 없음
- 기존 DINO가 cls 토큰만 썼다면 iBOT은 patch loss를 추가해 마스킹된 패치를 맞히게 함
- 다만 DINO의 cls loss는 다른 뷰끼리 비교하는 반면, MIM은 마스킹된 패치를 다뤄야 하므로 오직 두 개의 global view에만 적용

---

## 2. Preliminaries

### 2.1 Masked Image Modeling as Knowledge Distillation

**시각 토크나이저가 어려운 이유**
- BERT류의 성공은 언어를 의미 있는 단위로 토큰화할 수 있다는 데 기댐 (WordPiece 같은 것)
- 언어의 의미 단위는 단어 빈도 통계에서 자연스럽게 나오는 반면, 이미지는 연속적이라 시각적 의미를 그렇게 쉽게 뽑을 수 없음

**기존 MIM 의 두 갈래**
- **항등 사상을 토크나이저로** - 픽셀을 그대로 목표로. 의미 추상화에 약하고 고주파 디테일 모델링에 용량을 낭비
- **미리 학습한 토크나이저** - pretrained VAE(DALL-E VAE 등). 저수준 의미만 포착되고 다른 도메인으로 옮기기 어려움

두 번째 방식의 더 근본적인 문제는 다단계 파이프라인을 강요한다는 것이다. 목표 모델을 학습하기 전에 의미가 풍부한 토크나이저를 먼저 학습시켜야 한다. 그런데 시각적 의미를 획득하는 것은 어차피 두 단계의 공통 목표다. 그렇다면 따로 할 이유가 있나.

---

## 3. iBOT

### 3.1 Framework

**핵심 제안** - MIM을 토크나이저로부터의 knowledge distillation으로 정식화한다. 토크나이저 역할을 하는 twin teacher의 도움을 받아 distillation한다.

**입력 요소**
- **student(타겟 네트워크)** - 마스킹된 이미지
- **teacher(온라인 토크나이저)** - 마스킹되지 않은 원본 이미지
- **목표** - student가 마스킹된 각 패치 토큰을, 그 위치의 토크나이저 출력으로 복원

**이 방식으로 풀리는 문제**
- 클래스 토큰에 여러 각도의 이미지를 학습시켜 고수준 시각적 의미 포착
- teacher가 momentum update로 MIM과 공동 최적화됨 → 전처리 단계의 별도 학습이 필요 없음

**전체 흐름**
1. 원본 `x` 에서 augmentation으로 두 view `u`, `v` 생성. global view 2개가 기준이고 local view도 추가해 CLS loss 쪽에 사용
1. `u`, `v` 각각에 blockwise masking 적용 → masked view `û`, `v̂`
1. **Student** - `û`, `v̂` 를 받아 patch token 예측 분포 출력
1. Teacher(EMA 갱신) - 마스킹 안 된 `u`, `v` 를 받아 patch token target 분포 출력

**blockwise masking의 비율**
- prediction ratio `r` 은 마스킹할 토큰 비율. 기본 `r = 0.3`
- 학습 중에는 확률 0.5로 `r = 0`(마스킹 없음)을 섞고 나머지는 균등 분포에서 뽑음 → 항상 마스킹하지 않는다는 점이 특징

**MIM Loss (patch 수준)**
```text
L_MIM = -Σᵢ P_teacher(uᵢ) · log P_student(ûᵢ)
        (마스킹된 패치 위치 i 에 대해서만 합산)
```
- student가 본 `û` 의 마스킹 패치 예측이, teacher가 본 `u` 의 같은 위치 패치 출력을 따라가도록
- `u→û`, `v→v̂` 양쪽에 대해 계산 후 평균
- **두 개의 global view에만 적용** - local view는 MIM 대상이 아님

**CLS Loss (DINO 방식 그대로)**
- student의 CLS 토큰(masked view)과 teacher의 CLS 토큰(다른 view, 마스킹 없음)을 cross-view로 비교
- DINO의 self-distillation cross-entropy를 그대로 사용
- global과 local view 전체 조합에 대해 계산 → MIM과 달리 local도 포함

**최종 Loss** - 별도 가중치 없이 단순 합산
```text
L = L_MIM + L_CLS
```

### 3.2 Implementation

**Projection Head 공유** - CLS 토큰과 patch 토큰의 projection head를 공유한다. 파라미터를 따로 두지 않는다.

> DINOv2는 이 부분을 CLS용과 patch용으로 분리했다. iBOT loss를 가져오면서 명시한 차이점 중 하나이고, DINOv2 노트의 "MLP head 2개 생성" 부분과 대응된다.
> 흥미로운 건 방향이 규모에 따라 뒤집힌다는 점이다. 작은 규모에서는 공유가 낫고, 크게 키우면 분리가 낫다.

**Architecture**
- ViT-S/16, ViT-B/16, ViT-L/16, Swin-T 등 여러 backbone으로 실험
- online tokenizer가 별도 pretrain 없이 MIM objective와 동시에 학습된다는 게 핵심 → 기존 BEiT류가 고정 tokenizer(DALL-E VAE 등)를 쓰던 것과 대비

---

## 4. Experiment

- **4.1 ImageNet-1K 분류** - linear probing 82.3% (ViT-L/16)

- **4.2 Downstream** - COCO 검출·instance segmentation, ADE20K semantic segmentation에서 일관되게 향상

### 4.3 Properties of ViT trained with MIM

- **4.3.1 패치 토큰의 패턴 레이아웃** - 차량의 헤드라이트, 개의 귀 같은 부분 단위 패턴이 창발함
- 4.3.2 self-attention map의 판별적 부분 - 여러 head가 서로 다른 부분에 주목

- **4.3.3 강건성** - 배경 변경, 가림, 분포 밖 예제에 대해 더 강건

이렇게 학습된 특징이 강건성과 dense prediction 성능으로 이어진다는 것이 논문이 강조하는 부수 관찰이다.
