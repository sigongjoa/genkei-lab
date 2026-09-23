"""빠짐 검사 (Q8) — 키트 부품이 두 쪽 틀에서 한 축으로라도 빠지나 (09-23, README 「다음」 3 · art2real `조립.빠짐방향`).

    python 벤치/빠짐.py        키트 벤치가 남긴 부품(out/exe/<케이스>/부품/*.stl) -> out/빠짐.json · out/빠짐_*.png

축 방향 기둥마다 채워진 칸이 한 토막이면 그 축으로 빠진다(언더컷 없음). 칸 0.5 mm. 걸린 칸 몫 <= 0.2% 는 빠진다고 본다(칸 잡음).
원형사 규칙: 「한 방향으로 빠지면 통째, 안 빠지면 둘로」. 지금 키트는 부위(머리 · 몸통 · 팔 · 다리)로만 잘랐다 — 검사만 한다.

예측 (09-23, 돌리기 전에 커밋) — 사람형 35장:
  U1 탈형 불가 부품이 하나라도 있는 장이 50% 이상 (art2real 한 바퀴는 25 부품 중 10)
  U2 팔 · 다리 부품은 90% 이상 빠진다 (관 모양 로프트 · 층 타원 기둥)
  U3 몸통이 가장 많이 걸린다 (어깨 켜 · 치마 · 다보 구멍)
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
import 키트 as K            # noqa: E402

잡음 = 0.002


def 한장(c):
    d = os.path.join(A.OUT, "exe", c, "부품")
    out, 메시, 걸린점 = {}, {}, {}
    for f in sorted(os.listdir(d)):
        k = f[:-4]
        m = trimesh.load(os.path.join(d, f), force="mesh")
        몫, g, 걸림 = K.빠짐(np.asarray(m.vertices), np.asarray(m.faces))
        축 = [a for a, v in 몫.items() if v <= 잡음]
        out[k] = {"빠지는 축": 축, "걸린 몫": 몫}
        메시[k] = (np.asarray(m.vertices), np.asarray(m.faces))
        if not 축:                                                       # 가장 덜 걸리는 축의 걸린 칸 자리(그림용)
            a = min(몫, key=몫.get)
            zz, yy, xx = np.nonzero(걸림[a])
            lo = m.vertices.min(0)
            칸 = (m.vertices.max(0) - lo) / (np.array(g.shape[::-1]) - 2)
            걸린점[k] = lo + (np.stack([xx, yy, zz], 1) - 0.5) * 칸
    return out, 메시, 걸린점


def 그림(c, 결과, 메시):
    색표 = {k: ((90, 180, 110) if v["빠지는 축"] else (220, 70, 60)) for k, v in 결과.items()}
    im = Image.new("RGB", (560, 700), "white")
    im.paste(K.그리기(메시, (), 0.0, 색표=색표), (20, 50))
    d = ImageDraw.Draw(im)
    d.text((14, 10), "%s — 초록 빠짐 · 빨강 탈형 불가" % c[:34], fill="black", font=K._글꼴(16))
    y = 660
    나쁜 = ["%s(%s)" % (k, " ".join("%s %.0f%%" % (a, 100 * b) for a, b in v["걸린 몫"].items())) for k, v in 결과.items() if not v["빠지는 축"]]
    d.text((14, y), ("탈형 불가: " + ", ".join(나쁜))[:90] if 나쁜 else "모두 빠진다", fill=(200, 60, 50) if 나쁜 else (40, 150, 70), font=K._글꼴(13))
    return im


def main():
    벤치 = json.load(open(os.path.join(A.OUT, "키트벤치.json"), encoding="utf-8"))["기록"]
    기록 = {}
    for c, v in 벤치.items():
        if v["묶음"] == "도형":
            continue
        결과, 메시, _ = 한장(c)
        기록[c] = 결과
        그림(c, 결과, 메시).save(os.path.join(A.OUT, "빠짐_%s.png" % c))
        print("  %-34s %s" % (c[:34], " · ".join("%s%s" % (k, "○" if r["빠지는 축"] else "×") for k, r in 결과.items())), flush=True)
    장 = [any(not r["빠지는 축"] for r in v.values()) for v in 기록.values()]
    팔다리 = [bool(r["빠지는 축"]) for v in 기록.values() for k, r in v.items() if k.startswith(("팔", "다리"))]
    부위별 = {}
    for v in 기록.values():
        for k, r in v.items():
            부위별.setdefault(k, []).append(not r["빠지는 축"])
    불가율 = {k: round(float(np.mean(x)), 2) for k, x in 부위별.items()}
    값 = [(np.mean(장) >= 0.5, "탈형 불가 부품이 있는 장 %d/%d" % (sum(장), len(장))),
          (np.mean(팔다리) >= 0.9, "팔 · 다리 빠짐 %d/%d" % (sum(팔다리), len(팔다리))),
          (max(불가율, key=불가율.get) == "몸통", "부위별 탈형 불가율 %s" % 불가율)]
    예측 = [l.strip() for l in __doc__.splitlines() if l.strip().startswith("U")]
    print("\n예측")
    for 말, (맞, 글) in zip(예측, 값):
        print("  %s %s\n      %s" % ("○" if 맞 else "✗", 말, 글))
    json.dump({"기록": 기록, "예측": [{"예측": a, "맞음": bool(b), "값": t} for a, (b, t) in zip(예측, 값)]},
              open(os.path.join(A.OUT, "빠짐.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
