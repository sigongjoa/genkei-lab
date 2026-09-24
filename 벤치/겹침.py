"""겹침 — 옆 그림의 빈 곳도 파낸다 (09-24). 외부 평가 3회 원인 셋 중 「겹침 · 빈 공간」.

    python 벤치/겹침.py      -> out/겹침.json

  지금(층타원매끈): 켜마다 정면 구간 하나 = 폭, 옆 그림 한 줄 **전체**(맨 앞 ~ 맨 뒤) = 깊이. 다리가 앞뒤로 벌어지거나
  몸통 뒤에 틈이 있어도 막힌 덩어리가 된다.
  이번: 옆 그림 한 줄도 구간으로 나눠 정면 구간 × 옆 구간마다 타원 하나. 두 그림 어느 쪽과도 어긋나지 않는다
  (윤곽 교집합 안). 위아래 켜는 x 와 y 둘 다 겹치고 1대1 일 때 한 줄기로 잇는다.

예측 (09-24, 돌리기 전에 커밋):
  O1 안 되던 17장 F@2mm 중앙: 이번 - 지금 >= +0.02
  O2 잘 되던 대조 5장(사람형) F@2mm 중앙이 0.01 넘게 떨어지지 않는다

돌린 뒤 (09-24) — 둘 다 맞음:
  안되던 17장 F@2mm 중앙 0.455 -> 0.509 (+0.054). 17장 중 떨어진 것 0.
  로봇 0.581 -> 0.731 · 드릴 0.455 -> 0.509 · 로봇(다리 총) 0.459 -> 0.539 · 노움 0.641 -> 0.714 · 마녀 0.666 -> 0.738 ·
  메카 0.268 -> 0.317 · 검들기 0.619 -> 0.682. 그대로: 상자 둘 · 모자 · 종(옆 그림에 빈 곳이 없다 — 속 빔 · 꼴 문제).
  대조 5장 0.867 -> 0.867 (avatarsample_d +0.003, 나머지 같음). 그림 out/겹침.png — 로봇 팔 뒤 망토처럼 막히던 곳이 빠졌다.

exe 에 넣은 뒤 (09-24, 조각/그림.py 후보 「층 겹침」 · exe 다시 굽고 35장, 돌리기 전에 커밋):
  X1 35장 F@2mm 중앙 >= 0.727 (09-23 exe, out/지표_0923.json) — 계단 켜 대신 매끈이 되어도 잃지 않는다
  X2 안되던 17장 F@2mm 중앙 +0.03 이상 (exe 가 앱 점수로 층 겹침을 고르는가)
  X3 모든 장 exe 해시 = 파이썬 재생 해시

돌린 뒤 (09-24, out/지표_0924.json) — 셋 다 맞음:
  35장 F@2mm 중앙 0.727 -> 0.738 · 챔퍼 0.83 -> 0.87 · 새 각도 0.893 -> 0.886. 안되던 17장 F 0.462 -> 0.509. 해시 35/35.
  exe 가 층 겹침을 고른 곳 23장. 그런데 0.02 넘게 떨어진 7장(다리벌림 · 치마 · darkness · f · hairsample_female · knight Z6 ·
  chibi witch)은 모두 전에 「혼합」(팔 = 로프트) 이던 사람형 — 층 겹침은 T 포즈 팔까지 가로 켜로 쌓는다.
  -> 다음: 혼합 겹침(몸통 = 겹침 켜, 팔 = 로프트).

혼합 겹침 예측 (09-24, 돌리기 전에 커밋 · 파이썬 고르기로 흉내 -> 맞으면 exe):
  Y1 떨어졌던 7장 F@2mm 중앙 >= 그 7장의 09-23 중앙 - 0.01
  Y2 35장 F@2mm 중앙 >= 0.738
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


def 줄(case, 키=150.0):
    앞, 옆 = (G.마스크(os.path.join(시트, case, n + ".png")) for n in ("front", "side"))
    return G.층겹침(앞, 옆, 키)                                          # 09-24 exe 후보로 옮김 (조각/그림.py)


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
