set -e
P=./.venv-siglip/Scripts/python.exe
export PATTERN_DATA_ROOT="D:/CV/orig_4.4k_260519/orig_4.4k_260519"
export PYTHONIOENCODING=utf-8
# 두 인코더 모두 설명문 전체(약 256토큰)를 보게 맞춘다. 잘림 효과를 없애고 인코더만 남긴다.
echo "=== 1/2 SigLIP2 텍스트 타워 (64토큰 x 4창) ==="
$P experiments/text_encoder_ab.py --epochs 10 --only B_text_only \
   --win 64 --chunks 4 --out experiments/ab_siglip2.json
echo "=== 2/2 klue-roberta-large (256토큰 x 1창) ==="
$P experiments/text_encoder_ab.py --epochs 10 --only B_text_only \
   --text_model klue/roberta-large --win 256 --chunks 1 --out experiments/ab_klue.json
echo "=== AB DONE ==="
