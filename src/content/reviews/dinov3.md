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

**개관**
Dinov2 성능 좋으나, 근데 비규격화 데이터가 많거나 scale 다루는 거는 문제
**문제점**
라벨링 안된 데이터에서 쓸모있는 데이터를 얼마나 모았는 지 불분명하다
보통 train에서 cosine schedules 사용하는 데 큰 이미지 트레이닝 할 때 안좋다.
특징수가 점진적으로 감소해 초기 트레이닝 후에 비슷한 데이터만 다룬다.
→ 이런 문제는 ViT large 보다 더 큰 애들로 더 긴 학습에서 발생하고, dinov2의 경우에도 성능이 안좋다.
**해결**
downstream task에서 높은 성능을 보이면서 단일 frozen SSL backbone 을 일반적인 visual encoder로 쓸 수 있다.
**연구목표**
1. 다재다능한 기본 모델을 training 시키는 것
1. dense features에서 SSL model의 단점을 개선
1. 모델을 얼리고 써도 높은 성능 보이게 하는 것

**기존 문제점**
데이터를 많이 넣고 오래 학습시키면 사진 전체를 분류하는 능력이 좋아지지만 dense하게 보는 능력이 떨어지거나 뭉개지는 현상 존재
**해결책**
Gram Anchoring
기존 방식
- 화풍 변환 연구에서 쓰이던 Gram Matrix 개념을 차용
- 각 패치가 담고 있는 값을 feature로 사용
**Gram Matrix의 방식**
- 패치와 패치 사이의 유사도를 사용
- 패치 각각의 값을 바꾸는 건 괜찮아도 패치들 간의 관계 유사도를 유사하게 가져가게 하는 전략 사용
**작동 구조**
- 과거의 자기 자신을 teacher로 사용하여 Dense 성능이 낮아지기 전 초기 모델을 가져와서 dense 한 특징을 잘 잡아내게 규제한다.
- feature model과 student model에서 각각 패치 간 Gram matrix를 구한다.
  - feature 행렬(X)과 그 전치 행렬(X^T)를 곱하는 연산
  - 해당 연산을 통해 패치들 간 관계를 나타내는 행렬을 만든다(attention과 유사하지만 X*XT를 통해 그 값이 일정하게, 형이 비슷하게 유지되는 성질 사용)
- L{Gram} = ||Gram*{Student} - *Gram*{Teacher}* ||^2 식을 통해 teacher의 분포를 student가 닮아가게 하는 구조
**모델 스케일 및 Distillation**
- teacher 모델은 **7B 파라미터** ViT, DINOv2와 마찬가지로, 이 거대 teacher 하나로 여러 크기의 student
(ViT-S, B, L, 그리고 커스텀 S+, H+)를 동시에 distillation함.
- Distillation 단계에서는 Gram anchoring 없이 기존 objective만 사용 (작은 student는 애초에 dense feature 붕괴 문제가 덜하기 때문)
**고해상도 앵커링**
기본 학습 해상도는 256(효율성을 위해 작게 유지). 하지만 고해상도가 필요한 dense task(segmentation 등)를 위해, 학습 후반부에 고해상도 적응 단계를 추가함.
- Gram teacher는 student보다 높은 해상도로 이미지를 입력 받아 더 세밀한 patch-level feature를 뽑음.
- 이렇게 뽑은 teacher의 feature map을 student의 patch grid 크기에 맞춰 **다운샘플링**한 뒤 Gram matrix를 비교
- 고해상도 적응 단계에서는 global crop을 최대 768px까지 섞어서 10k iteration 정도 추가 학습
→ 이 방식으로 4k 해상도 입력에서도 안정적인 local feature 유지 가능
**추가 정의**
세밀 피처 잘잡는다. SSL 기반으로 훈련 저냑 수립하고 모델은 고수준 seg 성능을 가진다. 이것은 피처맵이 geo한 task에서의 높은 성능을 보이는 것을 가능하게 한다. #depth 추출이나 3D 매칭은 실시간성이 안나온다.
방대한 양의 데이터를 처리하면서도 세밀한(local) + 전역적(global) 특징을 동시에 잘 유지하기 어려운 문제가 있는데, 큰 모델일수록 collapse에 취약하기 때문. Gram anchoring이 이 문제를 해결해서, DINOv2 대비 더 dense하고 고해상도인 feature를 얻을 수 있게 됨.
- 결국 gram anchoring은 다량의 데이터를 처리함에도 불구하고 세밀한 + 전역적 특징을 처리 가능하게 해준다.
콜랩스 안빠지게 하는 전략. SSL에서 중요 특징점 Dinov2보다 덴스한 피처 따기 좋아졌고 높은 해상도를 가진다.
