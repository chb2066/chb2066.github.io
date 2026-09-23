# -*- coding: utf-8 -*-
"""이미지를 한 번만 디코딩해서 캐시로 굽는다.

매 epoch 마다 1024x1024 JPEG 4,400장을 열어 224로 줄이면 GPU 가 놀고 CPU 만 돈다.
  1) pixels.npy   224x224 uint8. 이미지 인코더를 푸는 조건에서 쓴다
  2) vfeat.npy    frozen vision tower 의 pooled 출력. 동결 조건에서는 매번 같은 값이라
                  미리 뽑아두면 학습이 몇 초로 끝난다
"""
import os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

REPO_TRAIN = os.environ.get(
    "KPEC_TRAIN_DIR",
    r"C:/Users/chb07/AppData/Local/Temp/claude/d--CV/36c80b59-0936-4007-8893-360d6ad9b884/scratchpad/repos/korean-pattern-emotion-classification/train",
)
sys.path.insert(0, REPO_TRAIN)
import dataset as D  # noqa: E402

MODEL = "google/siglip2-base-patch16-224"
OUT = Path("experiments/cache")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = D.load_records()
    ids = [r["id"] for r in recs]
    np.save(OUT / "ids.npy", np.array(ids))

    proc = AutoProcessor.from_pretrained(MODEL)
    size = 224

    px_path = OUT / "pixels.npy"
    if not px_path.exists():
        t = time.time()
        arr = np.lib.format.open_memmap(px_path, mode="w+", dtype=np.uint8,
                                        shape=(len(recs), size, size, 3))
        for i, r in enumerate(recs):
            im = Image.open(r["image_path"]).convert("RGB").resize((size, size), Image.BICUBIC)
            arr[i] = np.asarray(im, dtype=np.uint8)
            if (i + 1) % 500 == 0:
                print("  pixels %d/%d  %.0fs" % (i + 1, len(recs), time.time() - t), flush=True)
        arr.flush(); del arr
        print("pixels.npy 완료 %.0fs" % (time.time() - t), flush=True)

    vf_path = OUT / "vfeat.npy"
    if not vf_path.exists():
        dev = "cuda"
        m = AutoModel.from_pretrained(MODEL).vision_model.to(dev).eval()
        px = np.load(px_path, mmap_mode="r")
        mean = np.array(proc.image_processor.image_mean, dtype=np.float32)
        std = np.array(proc.image_processor.image_std, dtype=np.float32)
        feats = np.zeros((len(recs), m.config.hidden_size), dtype=np.float32)
        bs, t = 64, time.time()
        with torch.no_grad():
            for s in range(0, len(recs), bs):
                b = np.asarray(px[s:s + bs], dtype=np.float32) / 255.0
                b = (b - mean) / std
                x = torch.from_numpy(b).permute(0, 3, 1, 2).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = m(pixel_values=x).pooler_output
                feats[s:s + bs] = o.float().cpu().numpy()
                if (s // bs) % 20 == 0:
                    print("  vfeat %d/%d  %.0fs" % (s, len(recs), time.time() - t), flush=True)
        np.save(vf_path, feats)
        print("vfeat.npy 완료 %s %.0fs" % (feats.shape, time.time() - t), flush=True)
        del m
        torch.cuda.empty_cache()

    print("캐시 위치:", OUT.resolve())


if __name__ == "__main__":
    main()
