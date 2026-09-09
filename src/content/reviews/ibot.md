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

정리
iBOT은 크게 두 가지 아이디어 사용
1. DINO의 loss
  1. teacher와 student가 각각 다른 뷰를 보게 하고 student가 teacher의 확률 분포를 따라가게 만드는 구조 + EMA를 통해 teacher 업데이트량을 조절하는 구조
  1. teacher는 큰 뷰, student는 작은 뷰를 보고 학습을 하게 하여 loss를 통해 seg처럼 학습하게 함.
1. Ibot의 MIM
  1. teacher는 원본 이미지(masking x global o)를 보고 student 는 masking image(global + local global보다 local이 더 많은 2:8 or 2:10 정도, global 넣는 이유는 global없으면 학습 안됨, 부분 전체 관계 못봄 )를 본다.
  1. 기존 dino는 cls 토큰만 썼다면 얘는 patch loss를 사용해서 masking 된 패치를 맞추는 loss를 추가됨.
    1. MIM은 dino의 cls loss의 경우 다른 뷰를 통한 loss를 구하지만, 얘는 masking된 패치를 다루어야 하기 때문에  “MIM 목적함수(Loss)는 오직 두 개의 글로벌 뷰에만 적용” 한다.

### 서론

#### 기존 연구

**목표**
- MIM을 위해서는 시각적 의미 학습을 위해 온라인 표현 학습을 통해 학습이 이루어져야 한다.
- 토크나이저, 타겟 모델 둘 다 시각적 의미를 획득하는 것
  - pretrained VAE가 토크나이저로 제안, 저수준 의미 포착만 가능, 타 도메인 사용 불가능 등의 단점 존재

#### **iBOT 등장**

MIM을 토크나이저로부터 지식 증류해서 토크나이저로서 트윈 티처 도움 받아 distillation하는 것을 제안
**입력**
타겟 네트워크: masking 이미지
온라인 토크나이저: 원본 이미지
**목표**
타겟 네트워크가 마스킹된 각 패치 토큰을 그에 해당하는 토크나이저 출력으로 복원하게 하는 것
**해결 할 수 있는 것**
- 클래스 토큰에 대해 여러 각도의 이미지를 학습하게 하여 고수준 시각적 의미 포착
- momentum update를 통해 MIM과 공동으로 최적화되서 전처리 설정에서 별도 학습 필요 없음

### 방법론

**전체 흐름**
1. 원본 이미지 x에서 augmentation을 통해 두 개의 view(u, v) 생성(DINO처럼 global view 2개 기준, 여기에 local view도 추가로 생성해서 CLS loss 쪽에는 활용)
1. u, v 각각에 **blockwise masking** 적용 → masked view (û, v̂) 생성
1. Student network: masked view(û, v̂)를 입력받아 patch token들의 예측 분포 출력
1. Teacher network(online tokenizer, EMA로 업데이트): **마스킹 안 된 원본 view**(u, v)를 입력받아 patch token들의 target 분포 출력
**MIM Loss (Patch-level)**

```javascript
L_MIM = -Σᵢ P_teacher(uᵢ) · log P_student(ûᵢ) # (마스킹된 패치 위치 i에 대해서만 합산)
```

- Student가 본 masked view(û)의 각 마스킹 패치 예측이, teacher가 본 원본 view(u)의 같은 위치 패치 출력을 따라가도록 학습
- u→û, v→v̂ 양쪽에 대해 동일하게 계산 후 평균이 loss는 두 개의 global view에만 적용(local view는 MIM 대상 아님)
**CLS Loss (DINO 방식 그대로 차용)**
- Student의 CLS 토큰(masked view에서 나온)과, teacher의 CLS 토큰(다른 view에서 나온, unmasked)을 cross-view로 비교
- DINO의 self-distillation cross-entropy loss(L_CLS)를 그대로 사용
- Global view + local view 전체 조합에 대해 계산 (MIM과 달리 local view도 포함됨)
**최종 Loss**
L = L_MIM + L_CLS   (별도의 가중치(weighting) 없이 단순 합산)
** Projection Head 공유**
CLS 토큰과 patch 토큰에 대한 projection head를 공유해서 사용함(파라미터 따로 안 둠).
→ DINOv2에서는 이 부분을 CLS/patch용으로 분리했는데, 그게 DINOv2가 iBOT loss를 가져오면서 명시한 차이점 중 하나였음(DINOv2 노트에 있는 "MLP head 2개 생성" 부분과 대응됨).
**Architecture**
ViT-S/16, ViT-B/16, ViT-L/16, Swin-T 등 다양한 backbone으로 실험. Online tokenizer는 별도 pretrain 없이 MIM objective와 동시에 학습되는 게 핵심 특징 (기존 BEiT류가 미리 학습된 고정 tokenizer, 예: DALL-E VAE를 쓰던 것과 대비됨).
