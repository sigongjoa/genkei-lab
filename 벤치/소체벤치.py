"""대칭 기본형 소체 벤치 (09-25, 계획 단계 1).

    python 벤치/소체벤치.py      -> out/소체벤치.json · out/소체_*.png (입력 | 정답 | 켜부피 참고 | 소체 부품 색)

  35장 중 조각/소체.py 가 사람형으로 잡은 장만. 자: F@2mm 겉 · 챔퍼 · 짝 부품 거울 오차 + 그림(판단은 그림으로).

예측 (09-25, 돌리기 전에 커밋):
  S1 사람형으로 잡힌 장의 F@2mm 겉 중앙 >= 0.70 (켜부피 참고는 0.79 대)
  S2 모든 장 · 모든 짝 부품 거울 오차 < 0.001

돌린 뒤 (09-25) — 둘 다 맞음(고친 뒤):
  첫 판 F겉 0.665 — **벤치 버그**: 소체를 창 좌표로 옮길 때 메시 자신의 bbox 로 맞춰 배율이 어긋났다(기사 0.18) -> 그림 틀 그대로.
  미믹 거울 오차 0.005 — 팔다리 점을 0.1 mm 로 반올림해 거울 아닌 자세에서 짝 길이가 달랐다 -> 0.001 mm.
  22장(사람형으로 잡힌 장) F@2mm 겉 중앙 0.703 (켜부피 참고 0.845) · 거울 오차 최대 0.00001.
  잘 된 것: 마네킹 A · T 0.87~0.88 · base_male 0.85 · avatarsample_d/f 0.80. 짝 부품 모양은 완전히 같다.
  안 된 것: 차렷 · 팔 내린 자세는 팔을 몸통으로 잡는다(팔 길이 6.5 mm) · 검 들기 0.40 · 메카 · 미믹(사람 아님)을 사람형으로 잡음.
"""
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
import 지표 as J             # noqa: E402
import 그림 as G             # noqa: E402
import 키트 as K             # noqa: E402
import 소체 as S             # noqa: E402
from exe벤치 import 시트    # noqa: E402
from 넓은시트 import 잘린, 넓게  # noqa: E402

부품색 = {"머리": (231, 111, 81), "가슴": (170, 170, 185), "골반": (140, 140, 160),
        "윗팔": (86, 156, 214), "아래팔": (60, 120, 190), "허벅지": (106, 190, 120), "정강이": (70, 150, 90)}


def 색(이름):
    base = 부품색.get(이름.split(".")[0], (200, 200, 200))
    return tuple(int(c * (0.85 if 이름.endswith(".오") else 1.0)) for c in base)


def main():
    케이스 = list(json.load(open(os.path.join(A.OUT, "켜부피.json"), encoding="utf-8")))
    켜 = json.load(open(os.path.join(A.OUT, "켜부피.json"), encoding="utf-8"))
    기록, 판들 = {}, []
    for c in 케이스:
        뿌리 = 넓게 if c in 잘린 else 시트
        앞, 옆 = (G.마스크(os.path.join(뿌리, c, n + ".png")) for n in ("front", "side"))
        r = S.소체(앞, 옆, 150.0)
        if r is None:
            continue
        부 = S.부품들(r[0])
        # 창 좌표로 — **그림의 틀** 그대로(메시 크기로 맞추지 않는다: 09-25 첫 판은 메시 bbox 로 맞춰 발끝 · 머리끝이
        # 조금만 달라도 전체 배율이 어긋났다). 조각 좌표 x 는 몸 중심 기준(x0 를 뺐다), y 는 정면이 -y, z 는 바닥 0 · 키 150.
        x0 = r[1]["중심 x"]
        창 = lambda X: (np.asarray(X) + [x0, 0, 0]) * [1, -1, 1] / 150.0
        합 = trimesh.util.concatenate([trimesh.Trimesh(창(v), f, process=False) for v, f in 부.values()])
        g = A.정답(c)
        f겉 = J.F겉(J.mm메시(합.vertices, 합.faces), J.겉점(g)[0])[0]
        e = S.거울오차(부)
        기록[c] = {"F@2mm 겉": f겉, "켜부피 F@2mm 겉": 켜[c]["F@2mm 겉"], "부품": len(부), "거울 오차 최대": max(e.values()), **r[1]}
        print("%-32s 소체 F겉 %.3f (켜부피 %.3f) · 부품 %d · 거울 %.5f · 팔 %s" % (
            c[:32], f겉, 켜[c]["F@2mm 겉"], len(부), max(e.values()), r[1]["윗팔"]), flush=True)
        # 그림
        크기 = (260, 320)
        답 = g["메시"]
        곡 = trimesh.load(os.path.join(A.OUT, "켜부피", c, "메시.stl"), force="mesh")
        뒤 = np.array([1.0, -1.0, 1.0])
        셋 = [{"몸": (np.asarray(답.vertices) * 150 * 뒤, np.asarray(답.faces)[:, ::-1])},
              {"몸": (np.asarray(A.창(곡.vertices)) * 150 * 뒤, np.asarray(곡.faces)[:, ::-1])},
              {k: (창(v) * 150 * 뒤, np.asarray(f)[:, ::-1]) for k, (v, f) in 부.items()}]
        R = K._돌림()
        xy = np.vstack([(v @ R.T)[:, [0, 2]] for d in 셋 for v, _ in d.values()])
        범위 = (xy.min(0), xy.max(0))
        im = Image.new("RGB", (크기[0] * 4, 크기[1] + 26), "white")
        d = ImageDraw.Draw(im)
        fr = Image.open(os.path.join(뿌리, c, "front.png")).convert("RGBA")
        fr = Image.alpha_composite(Image.new("RGBA", fr.size, "white"), fr).convert("RGB")
        fr.thumbnail(크기)
        im.paste(fr, ((크기[0] - fr.width) // 2, 26))
        for i, (dd, 색표) in enumerate(zip(셋, [{"몸": (200, 200, 200)}, {"몸": (150, 185, 215)}, {k: 색(k) for k in 부}])):
            im.paste(K.그리기(dd, 크기=크기, 색표=색표, 범위=범위), (크기[0] * (i + 1), 26))
        d.text((6, 4), "%s   소체 F겉 %.3f · 켜부피 %.3f · 부품 %d · 거울 오차 %.4f" % (c, f겉, 켜[c]["F@2mm 겉"], len(부), max(e.values())),
               fill="black", font=K._글꼴(14))
        판들.append(im)
    med = float(np.median([v["F@2mm 겉"] for v in 기록.values()]))
    최대 = max(v["거울 오차 최대"] for v in 기록.values())
    print("%d장 · F겉 중앙 %.3f (켜부피 %.3f) · 거울 오차 최대 %.5f" % (len(기록), med, float(np.median([v["켜부피 F@2mm 겉"] for v in 기록.values()])), 최대))
    print("  %s S1 F겉 >= 0.70 · %s S2 거울 < 0.001" % ("○" if med >= 0.70 else "✗", "○" if 최대 < 1e-3 else "✗"))
    json.dump(기록, open(os.path.join(A.OUT, "소체벤치.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    for p in range(0, len(판들), 5):
        묶 = 판들[p:p + 5]
        out = Image.new("RGB", (묶[0].width, sum(i.height for i in 묶)), "white")
        y = 0
        for i in 묶:
            out.paste(i, (0, y))
            y += i.height
        out.save(os.path.join(A.OUT, "소체_%d.png" % (p // 5 + 1)))


if __name__ == "__main__":
    main()
