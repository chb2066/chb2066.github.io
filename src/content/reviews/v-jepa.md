---
title: V-JEPA
claim: 비디오에서도 픽셀이 아니라 임베딩 공간에서 마스킹된 시공간 영역을 예측하게 하면 강한 시각 표현이 학습된다.
tags: [Self-supervised, Video, JEPA]
tier: basic
date: 2025-07-16
draft: true
---

가시 세계를 인식하게 어떻게 학습시킬 수 있나? 정적 이미지 비디오로부터 시각적 잠재 표현 학습을 제안한다.

“첫번째 접근법은 텍스트 캡션을 사용해 잠재표현을 예측하는 VE를 학습하는 것이 있다. CLIP 처럼.
CLIP 모델 젤 큰 게 2B짜리인데 이건 2B 웹스크랩 이미지로부터 학습됐다. 이건 비디오나 이미지 단위 다운스트림 태스크에서 인상적인 성능을 보여줬다. 이것은 경량화가 적용된 태스크 특화 헤드를 사용하면서 가능했고 백본 얼려서 pt 모델에 대한 파인튜닝, 엔드투 엔드 학습이 요구되지 않았다. 그리고 이건 오디오 같은 데에서도 잘쓰였다. 인터넷 비디오랑 합쳐진 경우에도. 비디오 버트는 비디오 인코더가 마스킹된 영역의 표현을  측하게 텍스트 캡션을 사용해서 학습을 했다. 비슷하게 Vidoe CLIP 은 비디오 캡션에 대한 잠재 표현을 통해  대조 학습을 진행했다. 텍스트 인코더쓰면서. MERLOT ,VATT INTERNVIDEO 등은 이런걸 확장했다. Self-supervised 오디오 비디오 단위 로스를 적용하면서 VidieoCLIP을 확장했다. V-JEPA 성능 보여주겠다.”
→보통 이미지, 비디오 캡션 등을 사용하여 잠재 공간 혹은 마스크 공간을 예측하게 하는 경우, self-supervised한 오디오, 비디오 로스를 적용하는 경우가 많았고 이런 것들이 높은 성능을 냈다.

SSL 방법은 인간이 라벨링안하고 잠재 표현을 학습할 수 있다는 특징이 있다. 대신, 불변성 기반 사전학습은 VE를 손으로 만든 이미지 변화에 대한 불변성을 학습한다. 하지만 이건 상당한 양의 태스크 특화 유도 편향을 필요로 한다. 그리고 그것은 그것은 가능성을 제한한다. DAE는 손상된 입력을 재구성함으로써 학습 가능 잠재 표현을 대체할 수 있음을 보여준다. MAE는 Vidusal encoder-decoder가 입력에서 마스킹된 이미지 패치의 픽셀을 예측하게 한다. 미세조장할 때 MAE는 성능 좋다 하지만 MAE에 의해 만들어진 잠재 표현은 적응절차를 필요로 하고, 프로즌한 불변성 기반 생성된 표현에 미치지 못한다. 다른 태스크에서 masked image medeling은 날 픽셀보다 잠재 공간에서 예측하는 걸로 확정됐다.
DINOV2는 불변성 기반 로스랑 Masked image 모델링 로스 를 결합했고 SSL에서 경쟁력있다. CLIP처럼 텍스트 안써도 좋은 걸 보여준다. 하지만 정적 이미지 학습에서만이고 동적인 거에서 정보 수집한느 걸 기대하지는 못한다. V-JEPA는 한다.
→ maksing 기반 방ㅂ버들이 불변 잠재 공간을 다룰 때 좋은 성능을 보인 사례가 많다. 하지만 대부분 정적인 것에서 머물고 동적인 경우에 좋은 정보 수집 능력이 상대적으로 떨어진다. 하지만 V-JEPA에서 성공했다.

SSL from videos
비디오에서 불변 잠재 공간을 시간에 걸쳐 학습하게 하거나 다음 프레임에 대한 잠재표현을 계층 구조를 통해서 학습하게 하거나 seg를 사용해서 강한 잠재표현으로 이어질 수 있게 하거나 MAE의 접근법을 마스킹된 시공간 복셀을 예측하는 Encoder-decoder를 학습하게 해서 시공간 볼륨을 학습하게 하거나 했다.
사전 학습된 CLIP encoder의 frozen 잠재 공간에서 게산된 마스크 모델링 로스를 사용해서 모델학습을 했다.
근데 V-JEPA는 학습동안 온라인 학습으로 잠재 공간을 예측한다.
iMAE는 이미지랑 영상으로 학습시켰다. 한 논문에서는 MAE를 사용하는 계층적 트랜스포머 구조의 이점을 설명했고, 반면 다른 곳에서는 프레임 레벨 인코더를 사용해서 MAE로 확장하거나 cross-attention 기반 video-decoder를 사용하는 걸로 확장했다. MAE 기반 접근은 사전학습 동안 최소한의 편향을 만든다. 파인튜닝할 때 downstream task에서 높은  성능을 보인다. 하지만 V-JEPA는 더 좋은 잠재표현으로 이어지는 걸 보여준다.
→ SSL은 두 갈래로 나뉜다. 하나는 이미지 변형에 무관하게 특징을 뽑아내는 불변성 학습 기반이고 또 하나는 가려진 부분을 맞추는 재구성 기반 학습이다.

1. method
  1. JEPA
    1. 메인 아이디어는 jepa다 한 곳 보고 다른 곳 예측하는 방식, 네트워크 세 개 만들었다. 하나는 context에 대한 잠재 표현을 계산하는 context-encoder, 하나는 타겟의 잠재표현을 계산하는 target encoder, 나머지 하나는 그들 사이의 상대적 변화에 대한 정보를 통해 context-representation으로부터 target-representation을 예측하는 predictor이다. 세 가지 네트워크는 predictor랑 타겟 인코더 사이의 손실 최소화가 목적이다. 마스킹을 사용하는 V-JEPA를 설명한다. context-encoder, predictor는 마스킹된 비디오를 처리하고 마스킹 영역의 내용에 대한 예측을 출력한다. 이런 예측은 타켓 인코더 출력이랑 unmasked video를 비교한 L1 loss이다.
    →
    1. JEPA 기반이다.
    1. 네트워크 세 가지 쓴다.
      1. context-encoder
        1. input: 마스킹되고 남은 패치
        1. View: 비디오의 일부 정보
        1. update: predictor의 loss를 받아 학습
        1. output: 가시되는 부분에 대한 context representation을 출력하여 predictor로 전달
      1. target-encoder
        1. input: 원본 비디오 전체
        1. View: 비디오 전체 정보
        1. update: EMA로 업데이트
        1. output: 정답 벡터 출
      1. predictor
        1. input: context tokens, mask tokens
        1. outputs: 맥락 기반 정답 추측
        1. context representation으로부터 target-representation을 예측
    1. Loss는 L1 loss 사용, 타켓 인코더의 출력과 full video(masked token+unmasked toeknl)을 비교한 L1 loss 이다.
  > *[그림 자리 — Notion 원본에서 옮겨야 함]*

  d. input
  - 비디오에서 랜덤 시작점으로 연속 64프레임 클립 추출, temporal stride 4로 16프레임 균등 샘플링
  - 16프레임 클립에 3D conv (2×16×16 필터 d개) 적용 → 8×14×14×d 텐서
  - 3D positional embedding 추가 → flatten하여 1568×d 토큰 시퀀스
  - Multi-block masking 적용:
    - Short-range: 8개 블록의 union이 프레임 면적의 ~15% 커버
    - Long-range: 2개 블록의 union이 프레임 면적의 ~70% 커버
    - 종횡비: 0.75~1.5 랜덤
    - 공간 마스크를 시간축 전체에 반복 적용 (temporal mask ratio ≈ 1.0)
    - 최종 마스킹 비율: ~90%
  e. Patch-Level Loss
  1. context representation을 계산한다.
  1. V-JEPA loss 계산을 위해 video clip을 마스킹해서 context encoder에 줌으로써 context representation을 생성한다. 마스킹된 클립을 context encoder에 적용하는 것은 연속된 패치 잠재 표현이 타겟 영역을 예측하게 한다.
  1. V-JEPA predictor 네트워크는 입력으로 context encoder에 의해 생성된 토큰을 갖고 비디오 클립에서 잃어버린 영역을 예측한다. 그것은 학습 가능함 마스크 토큰으로 한다. 특히 마스크 토큰은 공유된 학습 가능 벡터와 3D sin-cos positional embedding의 합으로 파라미터화 된다.
  → predictor가 가려진 부분 예측, Target encoder가 전체 영상을 정답지로 만들고 L1 Loss로 둘의 차이를 계산해 학습하고 EMA를 정답지 기준을 안정화
  1. Multi-Mask Prediction
    1. 동일 비디오 기준 마스크를 0.15짜리 마스크 8번, 0.7 마스크 2번 개별적으로 predict하고 각 결과를 정답과 비교하고 오차들의 평균을 통해 loss 구성
  1. 붕괴 방지
    1. stop-gradient 사용, target encoder로 gradient 안가고 EMA로 업데이
