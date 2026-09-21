---
title: SAM
paper: Segment Anything
venue: ICCV 2023
link: https://arxiv.org/abs/2304.02643
claim: 프롬프트로 무엇을 자를지 지정하는 promptable segmentation을 사전학습 과제로 삼으면 zero-shot 전이가 가능한 분할 파운데이션 모델이 된다.
tags: [Segmentation, Foundation Model]
tier: basic
date: 2025-04-28
draft: true
---

[Segment Anything](https://arxiv.org/abs/2304.02643)

## Abstract

**문제**

- NLP는 웹 규모 사전학습 + prompting 으로 zero-shot 일반화를 얻었는데, segmentation에는 그에 해당하는 범용 사전학습 과제가 없었음
- 고정 클래스 집합을 분할하도록 학습하면 그 집합 밖으로 못 나감

**해결책**

- "무엇을 자를지"를 프롬프트로 받는 과제(promptable segmentation)를 사전학습 과제로 정의
- 이미지 1100만 장 · 마스크 10억 개 데이터셋을 data engine으로 구축
- 무거운 인코딩은 1회, 프롬프트마다는 경량 디코더만 → 실시간 상호작용

---

## 1. Introduction

논문이 스스로 던지는 세 질문이 구조를 만든다.
1. 어떤 과제가 zero-shot 일반화를 가능하게 하는가 → 2절
1. 그에 맞는 모델 구조는 무엇인가 → 3절
1. 그 과제와 모델을 뒷받침할 데이터는 무엇인가 → 4~5절

**foundation model과 downstream task**
- **foundation model** - 방대한 데이터로 사전학습된 범용 능력의 거대 모델
- **downstream task** - 그 모델로 해결하려는 구체적 작업. 의료 영상 분석, 자율 주행, 사진 편집 등

**NLP에서 배운 것**
- 웹 규모 사전학습 LLM은 제로샷 일반화를 가짐. 핵심 메커니즘은 prompt engineering
- 태스크를 텍스트 프롬프트로 표현해주면 파인튜닝 없이 일반화된 결과를 냄
- 이 능력은 데이터셋 크기·모델 크기·학습 비용이 커질수록 좋아짐

**비전에서의 시도**
- 대부분 image-text 쌍을 정렬하는 방식이었음
- SAM은 segmentation에서도 비슷한 수준의 범용 사전학습 과제를 정의하는 것이 목표

---

## 2. Segment Anything Task

**프롬프트란** - 전경/배경 점의 집합, 대략적인 박스, 마스크, 자유 형식 텍스트 등 이미지에서 무엇을 segment할지 나타내는 모든 정보

**과제의 정의** - 어떤 프롬프트가 주어져도 유효한 분할 마스크를 반환하는 것. 프롬프트가 모호해 여러 객체를 지칭할 수 있어도, 그중 적어도 하나에 대해서는 합당한 마스크를 출력해야 함

**NLP 개념을 가져온 이유**
1. 자연스러운 사전학습 알고리즘이 됨
1. prompting을 통해 downstream segmentation task로 zero-shot transfer가 가능해짐

**Pre-training** - 각 학습 샘플에 대해 순차적인 프롬프트를 시뮬레이션하고 정답과 예측을 비교

- interactive segmentation에서 가져왔지만 차이가 있음
- 기존은 충분한 사용자 입력 이후에 유효한 마스크를 점진적으로 예측
- SAM은 프롬프트가 아직 모호한 시점에도 곧바로 유효한 마스크를 예측하는 것이 목표
- 이렇게 학습해야 data engine이 요구하는 자동 주석처럼 모호함을 포함한 사용 사례에서도 잘 작동

기존 multi-task system과의 차이 - 이 구분이 중요하다.
- **multi-task system** - 고정된 여러 태스크를 수행. 학습 때 본 태스크와 테스트 때의 태스크가 동일
- **SAM** - 학습 시 보지 못한 새로운 태스크도 추론 시점에 수행 가능

이 특성 덕분에 SAM은 더 큰 시스템의 구성요소로 쓰일 수 있다. instance segmentation을 하려면 기존 객체 검출기와 결합하기만 하면 된다.

---

## 3. Segment Anything Model

**입력 요소** - 이미지 하나 + 프롬프트(점/박스/마스크/텍스트)

**Image encoder**
- 확장성과 강력한 사전학습을 고려해 MAE로 사전학습된 ViT 사용. 고해상도 입력을 처리하도록 최소한만 수정
- 이미지당 한 번만 실행되고, 프롬프트를 주기 전에 미리 적용 가능 → 이게 실시간 상호작용을 가능하게 하는 설계. 무거운 인코딩은 한 번, 이후 프롬프트마다의 처리는 가벼움

**Prompt encoder**

| 종류 | 무엇 | 어떻게 인코딩하나 |
|---|---|---|
| sparse | 점, 박스 | positional encoding 에 프롬프트 타입별 학습된 임베딩을 더함 |
| sparse | 자유 형식 텍스트 | CLIP의 텍스트 인코더를 그대로 사용 |
| dense | 마스크 | convolution으로 임베딩한 뒤 이미지 임베딩과 원소별로 더함 |

**Mask decoder** - 이미지 임베딩, 프롬프트 임베딩, 출력 토큰을 마스크로 매핑
- Transformer decoder 블록을 변형한 것에 동적 마스크 예측 head 를 붙인 구조
- prompt self-attention 과 양방향 cross-attention(프롬프트→이미지, 이미지→프롬프트)으로 모든 임베딩을 갱신
- 블록을 두 번 실행한 뒤 이미지 임베딩을 업샘플링하고, MLP가 출력 토큰을 매핑

양방향으로 attention한다는 게 중요하다. 프롬프트가 이미지를 읽을 뿐 아니라 이미지 임베딩도 프롬프트에 맞춰 갱신된다.

**모호성 처리**

- **문제** - 출력이 하나면 모호한 프롬프트에서 여러 유효한 마스크를 평균내버림. 셔츠의 한 점을 찍었을 때 셔츠인지, 사람인지, 셔츠의 무늬인지 알 수 없음
- **해결** - 한 프롬프트에 여러 개의 마스크를 출력
  - 3개면 대부분을 커버 → 중첩 마스크는 대개 깊어야 세 단계(전체 / 부분 / 부분의 부분)
  - 학습 시 손실이 가장 작은 마스크만 역전파 → 어느 것이 정답인지 모르므로 가장 잘 맞은 것만 반영
  - 순위를 매기기 위해 각 마스크에 신뢰도 점수(추정 IoU)를 함께 예측

---

## 4-5. Data Engine and Dataset

- 모델로 주석을 돕고, 그 주석으로 모델을 다시 학습시키는 순환 구조
- 결과 SA-1B - 이미지 1100만 장, 마스크 10억 개
- 프롬프트로 훈련될 수 있게 설계돼 새 이미지 분포와 새 태스크에 zero-shot transfer가 가능

---

## 7. Zero-Shot Transfer Experiments

자동 데이터셋 라벨링 외에 다섯 가지 샘플 태스크로 zero-shot 능력을 보인다.

- **7.1 단일 점 → 유효 마스크** - 23개 데이터셋에서 기존 최고 단일점 분할기보다 나음
- **7.2 edge detection** - edge를 예측하도록 학습한 적이 없는데도 동작
- 7.3 object proposals, 7.4 instance segmentation - 검출기 박스를 프롬프트로 넣는 것만으로 해결

- **7.5 text-to-mask** - 단순한 텍스트와 미묘한 텍스트 모두에서 동작
