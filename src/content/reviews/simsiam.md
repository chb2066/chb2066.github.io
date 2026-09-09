---
title: SimSiam
paper: Exploring Simple Siamese Representation Learning
venue: CVPR 2021
link: https://arxiv.org/abs/2011.10566
claim: EMA조차 없이 stop-gradient와 predictor만으로 collapse를 피할 수 있다.
take: gradient를 그대로 쓰면 업데이트가 너무 빨라지지 않나 싶었는데, 급격하지만 않다면 stop-gradient만으로 충분하다는 게 요지.
tags: [Self-supervised, Non-contrastive]
tier: basic
date: 2025-06-20
draft: true
---

**기존 방법**
기존 contrastive learning에서는 teacher를 사용하여 EMA 등의 기법을 사용해 업데이트 속도를 조절함.
BYOL에서는 branch 중 하나에만 predictor를두고 stop-gradient를 사용.
**SiaSiam 의 방법**
SimSiam은 BYOL과 비슷한 구조하게 stop-gradient를 사용하는 구조이지만, target과 online 파라미터가 동일하고 사실상 EMA만 제거한 것.

```javascript
Loss는 negative cosine similarity 사용:
D(p, z) = -(p/||p||₂) · (z/||z||₂)

p: predictor를 거친 online 브랜치 출력
z: predictor 없이 나온 target 브랜치 출력 (stop-gradient 적용)

양쪽 브랜치를 서로 바꿔서(symmetrized) 두 번 계산 후 평균:
L = 1/2 D(p1, stopgrad(z2)) + 1/2 D(p2, stopgrad(z1))
```

생각해볼만한 것:
기존 Contrastive 방법들은 target를 거의 업데이트 하지 않으면서 + collapse 방지를 위해 EMA로 조금씩 업데이트함. 하지만 위처럼 gradient를 그대로 사용하면 학습 업데이트가 너무 빨라지지 않나?
→ 업데이트가 급격하지 않다면 학습에 큰 영향을 주지 않고, tartget을 stop-gradient하는 것만으로도 충분하다.
