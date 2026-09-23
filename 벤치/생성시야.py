"""그림 한 장 + AI 가 만든 옆 그림 -> exe -> 채점 (09-23). 입력 결정: 서비스는 정면 한 장만 받는다.

    python 벤치/생성시야.py <케이스> <생성 시트 png>     -> out/생성시야/<케이스>/{side_gen.png, 결과.json, 비교.png}

  생성 시트에서 가장 왼쪽 덩어리 = 옆모습(Gemini 는 얼굴이 왼쪽 -> 뒤집어 정답 시트처럼 오른쪽 = 정면).
  같은 exe 로 (정답 옆) · (생성 옆) 두 번 돌려 챔퍼 · F@2mm · 새 각도를 나란히 잰다. 앞 그림은 둘 다 정답 시트 것.
  일관성: 키를 150 mm 로 맞춘 뒤 높이마다 옆 폭(깊이) mm 차이 — 3D 가 아니라 그림끼리.

예측 (09-23, 돌리기 전에 커밋) — avatarsample_d, Gemini 한 장:
  P1 깊이 차 중앙 <= 3 mm (눈으로는 실루엣이 비슷하다)
  P2 생성 옆 F@2mm >= 정답 옆 F@2mm - 0.10 · 챔퍼 중앙 증가 <= 0.5 mm
  P3 위에서 본 그림은 못 쓴다(팔 길이가 뒷모습의 절반) — 이번엔 옆만 쓴다
"""
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
import 지표 as J             # noqa: E402
import 그림 as G             # noqa: E402
from exe벤치 import exe로, 시트   # noqa: E402


def 옆뽑기(시트png, 뒤집기=True):
    a = np.asarray(Image.open(시트png).convert("RGB")).astype(int)
    m = ndimage.binary_fill_holes(ndimage.binary_closing(a.mean(-1) < 235, iterations=2))
    lab, n = ndimage.label(m)
    큰 = [i + 1 for i, s in enumerate(ndimage.find_objects(lab)) if (lab[s] == i + 1).sum() > 5000]
    i = min(큰, key=lambda k: np.nonzero(lab == k)[1].min())      # 가장 왼쪽
    옆 = lab == i
    return 옆[:, ::-1] if 뒤집기 else 옆


def 깊이줄(m, 층=150):
    """-> 높이 150 칸마다 폭(mm, 키 150). 빈 줄은 0."""
    ys, xs = np.nonzero(m)
    s = 150.0 / (ys.max() + 1 - ys.min())
    out = []
    for k in range(층):
        r = int(ys.max() - (k + 0.5) * (ys.max() + 1 - ys.min()) / 층)
        c = np.nonzero(m[r])[0]
        out.append((c.max() + 1 - c.min()) * s if len(c) else 0.0)
    return np.array(out)


def 재기(case, 폴더):
    m = trimesh.load(os.path.join(폴더, "메시.stl"), force="mesh")
    답 = A.정답(case)["메시"]
    r, _ = J.재기(J.mm메시(A.창(m.vertices), m.faces), J.mm메시(답.vertices, 답.faces))
    return r


def main(case, 시트png):
    D = os.path.join(A.OUT, "생성시야", case)
    os.makedirs(D, exist_ok=True)
    옆 = 옆뽑기(시트png)
    rgba = np.zeros(옆.shape + (4,), np.uint8)
    rgba[옆] = (0, 0, 0, 255)
    Image.fromarray(rgba).save(os.path.join(D, "side_gen.png"))
    Image.open(os.path.join(시트, case, "front.png")).save(os.path.join(D, "front.png"))
    Image.open(os.path.join(D, "side_gen.png")).save(os.path.join(D, "side.png"))

    진 = G.마스크(os.path.join(시트, case, "side.png"))
    d진, d생 = 깊이줄(진), 깊이줄(옆)
    일관 = {"깊이 차 중앙 mm": round(float(np.median(abs(d진 - d생))), 2),
            "깊이 차 90% mm": round(float(np.percentile(abs(d진 - d생), 90)), 2),
            "최대 깊이 정답 · 생성 mm": [round(float(d진.max()), 1), round(float(d생.max()), 1)]}

    결과 = {"일관성": 일관}
    for 이름, 뿌리 in (("정답 옆", 시트), ("생성 옆", os.path.join(A.OUT, "생성시야"))):
        폴더 = os.path.join(D, "exe_" + 이름.replace(" ", ""))
        exe로(case, 폴더, 뿌리)
        결과[이름] = 재기(case, 폴더)
        결과[이름]["1순위"] = json.load(open(os.path.join(폴더, "결과.json"), encoding="utf-8"))["후보"][0]["이름"]
    json.dump(결과, open(os.path.join(D, "결과.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(결과, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
