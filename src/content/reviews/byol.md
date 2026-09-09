---
title: BYOL
paper: Bootstrap Your Own Latent
venue: NeurIPS 2020
link: https://arxiv.org/abs/2006.07733
claim: negative pair 없이 positive pair만으로도, EMA target과 predictor 비대칭 구조가 collapse를 막아준다.
tags: [Self-supervised, Non-contrastive]
tier: basic
date: 2025-06-18
draft: true
---

기존 타 contrastive learning(moco, Simclr 등)은 negative pair, positive pair 등을 사용해서 학습을 진행했다면 얘는 positive pair 만을 사용해서 학습을 진행

**학습 방식**
동일 이미지를 다르게 aug한 두 view에 대해서 학습 진행함.
teacher는 predictor가 없고 Projection한 logit을 online으로 보냄.
online은 predictor를 MLP를 통해 진행하는 데 최종출력이 teacher의 차원 벡터가 predictor와 동일하여 loss를 통해서 두 값의 차이를 비교함.
**Loss**
MSE 사용, EMA 를 사용해서 teacher의 파라미터를 업데이트, student(online)는 loss로 인해 teacher분포와 최대한 가까워 지는 방향으로 학습됨.
**collapse 방지**
target은 EMA로 천천히 업데이트하기 때문에 큰 영향을 주지 않고 조금씩 변하기 때문에 다양한 정보 사용 가능.
