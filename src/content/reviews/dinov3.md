---
title: DINOv3
paper: DINOv3
venue: arXiv 2025
link: https://arxiv.org/abs/2508.10104
claim: 오래 학습할수록 무너지는 dense feature를 Gram anchoring으로 붙잡으면, 얼린 백본만으로 dense task까지 커버한다.
tags: [Self-supervised, Vision Backbone]
tier: basic
date: 2025-07-02
draft: true
---

## 주요 전략
1. 패치 자체가 아니라 패치 간 Gram 행렬을 과거 체크포인트에 정박시켜 dense feature 퇴화 방지함.
2. 고해상도 적응 단계를 추가해 얼린 백본으로 segmentation, depth estimation 수행함.

## 배경 지식

### DINOv2에서 남은 문제

DINOv2는 성능이 좋다. 그런데 비정규화 데이터가 많거나 규모를 다루는 데서 문제가 있다.

- 라벨링되지 않은 데이터에서 **쓸모 있는 데이터를 얼마나 모았는지 불분명함.**
- 보통 학습에 cosine schedule을 쓰는데, **큰 이미지를 학습할 때는 이게 좋지 않음.**
- 학습이 진행되면서 **다루는 특징의 다양성이 점차 줄어듦.** 초기 학습 이후에는 비슷한 데이터만 처리하게 됨.

이런 문제는 **ViT-Large보다 큰 모델을 더 오래 학습할 때** 나타나고, DINOv2도 예외가 아니다.

### 연구 목표

1. 다재다능한 기본 모델을 학습시킴.
2. **dense feature에서 SSL 모델의 약점을 개선함.**
3. 모델을 얼린 채로 써도 높은 성능이 나오게 함.

목표는 결국 하나다. **단일 frozen SSL backbone을 범용 visual encoder로** 쓸 수 있게 만드는 것이다.

## 핵심 문제 — dense feature의 퇴화

**데이터를 많이 넣고 오래 학습시키면 사진 전체를 분류하는 능력은 좋아진다. 그런데 dense하게 보는 능력이 떨어지거나 뭉개진다.**

논문은 이를 **알려져 있지만 해결되지 않았던 문제**로 명시한다. 원인은 두 목표가 서로 상충한다는 데 있다.

- 고수준 이해(전체 이미지가 무엇인가)를 잘하려면 세부를 버리고 추상화해야 함.
- dense feature 품질(각 패치가 무엇인가)을 지키려면 세부를 유지해야 함.

큰 모델일수록 오래 학습하면 전자로 기울고, **collapse에 취약**해진다.

## Gram Anchoring

### 아이디어의 출처

화풍 변환 연구에서 쓰이던 **Gram Matrix** 개념을 차용한다. 각 패치가 담고 있는 값을 feature로 사용한다.

### Gram Matrix가 재는 것

패치와 패치 사이의 유사도다. 여기서 전략이 나온다.

> **패치 각각의 값이 바뀌는 것은 괜찮다. 다만 패치들 사이의 관계는 비슷하게 유지되게 한다.**

### 작동 구조

- **과거의 자기 자신을 teacher로 사용함.** dense 성능이 낮아지기 전의 초기 모델을 가져와, dense한 특징을 잘 잡아내도록 규제함.
- teacher 모델과 student 모델에서 각각 패치 간 Gram matrix를 구함.
  - feature 행렬 `X`와 그 전치 `Xᵀ`를 곱하는 연산임.
  - 이 연산으로 패치들 사이의 관계를 나타내는 행렬이 만들어짐. attention과 비슷하지만, `X·Xᵀ` 형태라 값이 일정하게, 형태가 비슷하게 유지되는 성질을 활용함.
- 다음 식으로 teacher의 분포를 student가 닮아가게 함.

```text
L_Gram = || Gram_student − Gram_teacher ||²
```

### 이 방법이 통하는 이유

표현 자체는 계속 발전해도 된다. 전역 성능은 계속 오르게 두고, **잃어가는 축(패치 간 기하 관계)만 과거 시점에 정박**시키는 것이다. 규제 대상을 좁힌 것이 핵심이다.

## 모델 스케일과 Distillation

- teacher 모델은 **7B 파라미터 ViT**다.
- DINOv2와 마찬가지로 이 거대한 teacher 하나로 여러 크기의 student(ViT-S, B, L, 그리고 커스텀 S+, H+)를 **동시에 distillation**함.
- **Distillation 단계에서는 Gram anchoring을 쓰지 않음.** 기존 objective만 씀. 작은 student는 애초에 dense feature 붕괴 문제가 덜하기 때문임.

## 고해상도 앵커링

기본 학습 해상도는 효율성을 위해 **256**으로 작게 유지한다. 하지만 segmentation처럼 고해상도가 필요한 dense task를 위해 학습 후반부에 고해상도 적응 단계를 추가한다.

- **Gram teacher가 student보다 높은 해상도로 이미지를 받아** 더 세밀한 patch-level feature를 뽑음.
- 이렇게 뽑은 teacher의 feature map을 student의 patch grid 크기에 맞춰 **다운샘플링**한 뒤 Gram matrix를 비교함.
- 고해상도 적응 단계에서는 global crop을 최대 768px까지 섞어서 10k iteration 정도 추가 학습함.

이 방식으로 **4k 해상도 입력에서도 안정적인 local feature가 유지**된다.

## 정리

DINOv3가 푼 문제는 "**오래 학습할수록 dense feature가 무너진다**"이고, 해법은 **잃어가는 축만 골라 과거의 자기 자신에 정박시키는 것**이다.

여기서 옮겨갈 만한 재료가 두 개 들어 있다.

**첫째, 특징이 아니라 관계 구조를 규제하기.** 표현은 자유롭게 움직이되 기하는 보존한다. 어떤 표현 학습에도 붙일 수 있고, 보존하려는 성질을 쌍별 내적으로 쓸 수만 있으면 된다.

**둘째, 성능 지표가 오르는 동안 조용히 나빠지는 다른 축이 있다는 진단틀.** 전체 분류 정확도만 보고 있으면 dense 품질이 떨어지는 것을 못 본다. 파인튜닝 중 잃는 능력 — 안전성, 일반화, 캘리브레이션 — 에 그대로 대응하는 문제다.

다만 가정도 함께 온다. **과거 체크포인트가 그 성질에 대해 실제로 더 나아야 하고**, 어느 시점을 teacher로 삼을지 정하는 기준이 필요하다.
