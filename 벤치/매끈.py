"""계단 없애기 — 층 타원 켜를 평평한 기둥으로 쌓지 말고 높이마다 잰 수평 단면을 잇는다(dsl `층쌓기`) (09-23).

    python 벤치/매끈.py        사람형 35장, exe 가 고른 종류(혼합 · 층 타원)마다 계단판 대 매끈판 -> out/매끈.json · out/매끈_*.png

자는 IoU 말고(사용자: 다른 세션에서 IoU 가 문제로 확인됐다): 챔퍼 · F@2mm · 새 각도 실루엣 · 표면각.
개발 두 장(T포즈 · avatarsample_d): 표면각 중앙 0.0 · 2.3° → 0.0 · 4.3° (정답 5.5 · 4.1°) · 90% 90° → 17~47° · F -0.005~-0.009.

예측 (09-23, 돌리기 전에 커밋) — 사람형 35장:
  S1 표면각 중앙이 정답에 가까워진다: |우리 - 정답| 의 중앙이 줄어든다
  S2 F@2mm 중앙이 0.01 넘게 떨어지지 않는다
  S3 챔퍼 중앙은 +0.05 mm 안 · 새 각도 중앙은 -0.005 안
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "조각")]
import 어댑터 as A           # noqa: E402
import 그림 as G            # noqa: E402
import dsl as D             # noqa: E402
import 지표 as Z            # noqa: E402
import 키트 as K            # noqa: E402

짝 = {"혼합": (G.혼합, G.혼합매끈), "층 타원": (G.층타원, G.층타원매끈)}


def main():
    벤치 = json.load(open(os.path.join(A.OUT, "키트벤치.json"), encoding="utf-8"))["기록"]
    기록 = {}
    for c, v in 벤치.items():
        if v["묶음"] == "도형" or v["1순위"] not in 짝:
            continue
        시트 = os.path.join(A.원화3d, "out", "시트", c)
        앞, 옆 = G.마스크(os.path.join(시트, "front.png")), G.마스크(os.path.join(시트, "side.png"))
        답 = A.정답(c)["메시"]
        답m = Z.mm메시(답.vertices, 답.faces)
        r, 메시 = {}, {}
        for 판, f in zip(("계단", "매끈"), 짝[v["1순위"]]):
            V, F = D.실행(f(앞, 옆, 150.0))
            r[판], _ = Z.재기(Z.mm메시(A.창(V), F), 답m)
            메시[판] = (V, F)
        기록[c] = r
        if c in ("avatarsample_d", "base_female", "knight_Warrior_Z6ZUtm6kc1", "마네킹_A포즈"):
            im = Image.new("RGB", (1080, 700), "white")
            d = ImageDraw.Draw(im)
            for j, 판 in enumerate(("계단", "매끈")):
                im.paste(K.그리기({"몸통": 메시[판]}, 크기=(520, 640)), (j * 540 + 10, 50))
                d.text((j * 540 + 20, 12), "%s — 표면각 중앙 %.1f° (정답 %.1f°) · F@2mm %.3f · 챔퍼 %.2f mm" % (
                    판, r[판]["표면각 우리"][0], r[판]["표면각 정답"][0], r[판]["F@2mm"], r[판]["챔퍼 중앙"]), fill="black", font=K._글꼴(15))
            im.save(os.path.join(A.OUT, "매끈_%s.png" % c))
        print("  %-34s 표면각 %.1f→%.1f (정답 %.1f) · F %.3f→%.3f · 챔퍼 %.2f→%.2f · 새각도 %.3f→%.3f" % (
            c[:34], r["계단"]["표면각 우리"][0], r["매끈"]["표면각 우리"][0], r["계단"]["표면각 정답"][0],
            r["계단"]["F@2mm"], r["매끈"]["F@2mm"], r["계단"]["챔퍼 중앙"], r["매끈"]["챔퍼 중앙"],
            r["계단"]["새 각도 중앙"], r["매끈"]["새 각도 중앙"]), flush=True)
    med = lambda 판, f: float(np.median([f(x[판]) for x in 기록.values()]))
    차 = lambda 판: med(판, lambda x: abs(x["표면각 우리"][0] - x["표면각 정답"][0]))
    값 = [(차("매끈") < 차("계단"), "|우리-정답| 중앙 %.2f° → %.2f°" % (차("계단"), 차("매끈"))),
          (med("매끈", lambda x: x["F@2mm"]) >= med("계단", lambda x: x["F@2mm"]) - 0.01,
           "F@2mm %.3f → %.3f" % (med("계단", lambda x: x["F@2mm"]), med("매끈", lambda x: x["F@2mm"]))),
          (med("매끈", lambda x: x["챔퍼 중앙"]) <= med("계단", lambda x: x["챔퍼 중앙"]) + 0.05 and
           med("매끈", lambda x: x["새 각도 중앙"]) >= med("계단", lambda x: x["새 각도 중앙"]) - 0.005,
           "챔퍼 %.2f → %.2f mm · 새 각도 %.3f → %.3f" % (med("계단", lambda x: x["챔퍼 중앙"]), med("매끈", lambda x: x["챔퍼 중앙"]),
                                                       med("계단", lambda x: x["새 각도 중앙"]), med("매끈", lambda x: x["새 각도 중앙"])))]
    예측 = [l.strip() for l in __doc__.splitlines() if l.strip().startswith("S")]
    print("\n예측 (%d 장)" % len(기록))
    for 말, (맞, 글) in zip(예측, 값):
        print("  %s %s\n      %s" % ("○" if 맞 else "✗", 말, 글))
    json.dump({"기록": 기록, "예측": [{"예측": a, "맞음": bool(b), "값": t} for a, (b, t) in zip(예측, 값)]},
              open(os.path.join(A.OUT, "매끈.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
