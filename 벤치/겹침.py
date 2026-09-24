"""겹침 — 옆 그림의 빈 곳도 파낸다 (09-24). 외부 평가 3회 원인 셋 중 「겹침 · 빈 공간」.

    python 벤치/겹침.py      -> out/겹침.json

  지금(층타원매끈): 켜마다 정면 구간 하나 = 폭, 옆 그림 한 줄 **전체**(맨 앞 ~ 맨 뒤) = 깊이. 다리가 앞뒤로 벌어지거나
  몸통 뒤에 틈이 있어도 막힌 덩어리가 된다.
  이번: 옆 그림 한 줄도 구간으로 나눠 정면 구간 × 옆 구간마다 타원 하나. 두 그림 어느 쪽과도 어긋나지 않는다
  (윤곽 교집합 안). 위아래 켜는 x 와 y 둘 다 겹치고 1대1 일 때 한 줄기로 잇는다.

예측 (09-24, 돌리기 전에 커밋):
  O1 안 되던 17장 F@2mm 중앙: 이번 - 지금 >= +0.02
  O2 잘 되던 대조 5장(사람형) F@2mm 중앙이 0.01 넘게 떨어지지 않는다
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
import 그림 as G             # noqa: E402
from 틀빌더 import 재기, 시트  # noqa: E402

안되던 = json.load(open(os.path.join(A.OUT, "틀라벨", "정답.json"), encoding="utf-8"))
안되던 = [v["케이스"] for v in 안되던.values()]
대조 = ["avatarsample_d", "base_male", "base_female", "마네킹_차렷", "마네킹_T포즈"]


def 켜재기(앞, 옆, fa, fo, 층수=60):
    """켜마다 (z0, z1, [(a, b, y, 깊이)]) 아래→위. a, b = 정면 화소 구간 · y, 깊이 = 옆 구간 mm."""
    rows = np.nonzero(앞.any(1))[0]
    경계 = np.linspace(rows.min(), rows.max() + 1, 층수 + 1)
    켜들 = []
    for r0, r1 in zip(경계[:-1], 경계[1:]):
        r = int((r0 + r1) / 2)
        z0, z1 = (fa.bot - r1) * fa.s, (fa.bot - r0) * fa.s
        ro = int(np.clip(round(fo.bot - (z0 + z1) / 2 / fo.s - 0.5), 0, 옆.shape[0] - 1))
        옆조각 = G._줄조각(옆[ro])
        상자 = [(a, b, -((c + d) / 2 - fo.c) * fo.s, (d - c) * fo.s) for a, b in G._줄조각(앞[r]) for c, d in 옆조각]
        if 상자:
            켜들.append((z0, z1, 상자))
    return 켜들[::-1]


def _겹(p, q):
    return min(p[1], q[1]) > max(p[0], q[0]) and abs(p[2] - q[2]) < (p[3] + q[3]) / 2


def 줄기(켜들):
    """x · y 둘 다 겹치고 1대1 이면 잇는다 (그림.켜줄기 와 같은 규칙, 겹침만 2D)."""
    줄기들, 열린 = [], []
    for i, (_, _, 상자) in enumerate(켜들):
        새 = []
        for g in 상자:
            닿 = [줄 for 줄 in 열린 if _겹(g, 줄[-1][1])]
            if len(닿) == 1 and sum(_겹(gg, 닿[0][-1][1]) for gg in 상자) == 1:
                닿[0].append((i, g))
                새.append(닿[0])
                continue
            줄 = [(i, g)]
            줄기들.append(줄)
            새.append(줄)
        열린 = 새
    return 줄기들


def 줄(case, 키=150.0):
    앞, 옆 = (G.마스크(os.path.join(시트, case, n + ".png")) for n in ("front", "side"))
    fa, fo = G.틀(앞, 키), G.틀(옆, 키)
    켜들 = 켜재기(앞, 옆, fa, fo)
    토막 = []
    for 줄 in 줄기(켜들):
        z, x, y, w, d = [], [], [], [], []
        for n, (i, (a, b, yy, 깊이)) in enumerate(줄):
            z0, z1, _ = 켜들[i]
            for h in ([z0] if n == 0 else []) + [(z0 + z1) / 2] + ([z1 + 0.01] if n == len(줄) - 1 else []):
                z.append(round(float(h), 2)); x.append(round(float(((a + b) / 2 - fa.c) * fa.s), 1)); y.append(round(float(yy), 1))
                w.append(round(float((b - a) * fa.s), 1)); d.append(round(float(깊이), 1))
        토막.append("층쌓기(z=%s, x=%s, y=%s, w=%s, d=%s)" % (z, x, y, w, d))
    return " + ".join(토막)


def main():
    기록 = {}
    for c in 안되던 + 대조:
        앞, 옆 = (G.마스크(os.path.join(시트, c, n + ".png")) for n in ("front", "side"))
        r = {"지금": 재기(c, G.층타원매끈(앞, 옆, 150.0)), "이번": 재기(c, 줄(c))}
        기록[c] = r
        print("%-32s 지금 F %.3f 챔퍼 %.2f · 이번 F %.3f 챔퍼 %.2f" % (c[:32], r["지금"]["F@2mm"], r["지금"]["챔퍼 중앙"],
                                                                r["이번"]["F@2mm"], r["이번"]["챔퍼 중앙"]), flush=True)
    med = lambda k, cs: float(np.median([기록[c][k]["F@2mm"] for c in cs]))
    a0, a1, b0, b1 = med("지금", 안되던), med("이번", 안되던), med("지금", 대조), med("이번", 대조)
    print("안되던 17장 F 중앙 %.3f -> %.3f · 대조 5장 %.3f -> %.3f" % (a0, a1, b0, b1))
    print("  %s O1 +0.02 이상\n  %s O2 대조 -0.01 안쪽" % ("○" if a1 - a0 >= 0.02 else "✗", "○" if b1 - b0 >= -0.01 else "✗"))
    json.dump(기록, open(os.path.join(A.OUT, "겹침.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
