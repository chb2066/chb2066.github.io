---
title: U-Net
paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
venue: MICCAI 2015
link: https://arxiv.org/abs/1505.04597
claim: 대칭 encoder-decoder에 skip connection을 넣어, 적은 의료 영상 데이터로도 경계까지 복원하는 segmentation이 가능하다.
tags: [Segmentation, Medical]
tier: basic
date: 2025-03-17
draft: true
---

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## Abstract

**문제**
딥러닝 segmentation은 라벨된 학습 데이터가 수천 장 필요한데, 의료 영상은 전문가가 라벨링해야 해서 그만큼 모을 수 없음
게다가 세포처럼 같은 클래스의 인접 객체를 개별로 분리해야 함

**해결책**
대칭 U자형 encoder-decoder에 skip connection을 넣어 다운샘플링에서 잃은 공간 정보를 복원
data augmentation을 적극 사용하고, 경계에 가중치를 주는 weight map으로 붙어 있는 객체를 분리

---

## 1. Introduction

**의료 영상이라는 조건이 설계를 규정한다**
- 데이터가 적음 → 전문가 라벨링이 필요해 수천 장을 모으기 어려움
- 같은 클래스의 인접 객체를 분리해야 함 → 세포처럼 맞닿아 있는 것들을 개별로 구분

**각 제약에 대한 대응**
- 데이터 부족 → data augmentation 적극 사용
- 경계 분리 → weight map으로 경계 강조

**구조적 선택**
- Encoder-Decoder 구조로 전역 정보와 세부 정보를 함께 활용
- Skip connection 사용

---

## 2. Network Architecture

**세 가지 특징**
1. **대칭 U자형 구조**
1. FC Layer 없이 conv 연산만 사용 → flatten이 없으므로 공간 정보가 소실되지 않음
1. 다운샘플링 경로의 고해상도 특징을 업샘플링 경로와 결합해 세밀한 정보 복원

Encoder에서 전역적 문맥을 학습하고, Decoder에서 해상도와 경계를 복원한다.

**Encoder - Contraction Path (다운샘플링)**
1. Conv 3×3 + ReLU 2번
1. Max pool 2×2

**Decoder - Expanding Path (업샘플링)**
1. feature map을 2×2 up-conv로 업샘플링
1. **skip connection** - 다운샘플링에서 나온 feature map을 업샘플링된 것에 concat
   - 두 이미지 크기가 다르므로 업샘플링된 쪽에 패딩을 더해 맞춤
1. Conv 3×3 + ReLU 2번. 첫 conv에서 채널 수를 맞춰 세부 정보 복원
1. 1×1 Convolution으로 최종 클래스 개수만큼 채널 수 조정

**Output** - softmax로 픽셀 단위 확률값을 출력

### Skip connection이 필요한 이유

**Encoder-Decoder 구조의 문제**
- 다운샘플링을 반복하면서 feature map이 축소 → 세밀한 정보와 픽셀 위치 정보가 손실
- conv 과정에서 경계 정보 손실 발생
→ 업샘플링할 때 경계·위치·세밀한 정보가 부족해 흐릿한 결과

**해결**
- 인코더의 고해상도 특징을 디코더의 업샘플된 출력과 결합하고, 이후 conv layer가 조합
- 단순히 더하는 게 아니라 조합할 기회를 주는 것이 요점

이 구조가 주는 것은 디코더가 "무엇을"과 "어디에"를 나눠 풀 수 있게 된다는 점이다. 무엇인지는 깊은 층의 저해상도 특징이 알고, 어디인지는 얕은 층의 고해상도 특징이 안다.

---

## 3. Training

**Overlap-Tile 전략**

- GPU 메모리 한계로 이미지 전체를 한 번에 처리할 수 없음 → 큰 입력 타일로 나눠 처리
- 문제: 타일 경계 부근을 예측하려면 그 바깥 문맥이 필요한데 이미지 밖이라 없음
- 해결: 경계를 거울처럼 반사(mirroring)해서 외삽
- 큰 타일을 쓰면 더 많은 문맥 정보를 얻을 수 있음

**손실 함수**
- 픽셀 단위 Softmax + Cross Entropy
- **Weight Map을 추가해 경계 강조**

![그림](/img/unet/03.png)

- weight map의 목적: 같은 클래스에 속하는 인접 객체가 붙어 있을 때 그 사이의 경계를 학습하도록 추가 가중치 부여
- 세포 두 개가 맞닿아 있으면 그 접촉선의 픽셀들이 가장 중요한 픽셀이 됨

![그림](/img/unet/04.png)

**가중치 초기화**
- feature map이 단위 분산을 유지하도록 조정 → He Initialization, 표준편차 `√(2/N)` 로 샘플링

### 3.1 Data Augmentation

- 데이터가 적으므로 augmentation이 선택이 아니라 필수
- 특히 elastic deformation - 조직이 변형되는 실제 양상을 흉내 내는 변형이라 이 도메인에서 값이 큼

---

## 4. Experiments

- **ISBI 세포 추적 챌린지 2015 우승** - 두 개 범주 모두에서 큰 격차
- 전자현미경 스택의 신경 구조 분할에서 당시 최고 성능
- NVidia Titan GPU에서 512×512 이미지 분할에 1초 미만

데이터가 적은 상황에서 augmentation만으로 이 성능이 나왔다는 점이 함께 강조된다.

---

## 정리

U-Net에서 가져갈 것은 "다운샘플링에서 잃는 것을 옆길로 넘겨준다"는 구조다.

깊이가 깊어질수록 의미는 풍부해지지만 위치는 사라진다. 이 상충을 해결하는 방법은 두 정보를 다른 경로로 전달해서 나중에 합치는 것이다. 하나의 경로로 모두 전달하려 하면 어느 쪽이든 손해를 본다.

이 구조는 의료 영상을 훨씬 넘어서 퍼졌다. 그리고 diffusion 모델의 백본으로도 오래 쓰였다 - DiT가 그걸 Transformer로 교체하기 전까지.

다만 옮길 때 주의할 점이 있다. 인코더 특징이 디코더에 유용한 형태여야 한다. 도메인이 다르면 skip이 오히려 잡음을 넘겨주는 통로가 된다.

