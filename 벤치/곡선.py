"""다루는 건 전부 곡선 (09-24 사용자: 「아예 다루는 게 곡선이어야」 · 「F 가 떨어져도 형태가 비슷하면 된다」).

    python 벤치/곡선.py [n]      앞에서부터 n 장 더(이어서) -> out/곡선/<케이스>/{메시.stl, 결과.json} · out/곡선.json

  후보에서 계단(혼합 · 층 타원)을 빼고 매끈한 판(혼합 겹침 · 층 겹침 · 부위 둘)을 dsl 「부드럽게」로 감쌌다:
  조각마다 부호 거리장 -> 다항식 부드러운 최솟값(반경 3 mm) -> manifold level_set. 이음매(겨드랑이 · 가랑이 · 목)가 둥글다.
  exe 대신 파이썬 고르기(그림.고르기 — exe 와 같은 함수)로 35장. 잘렸던 4장은 넓은 시트. 판단은 갤러리 그림으로.

예측 (09-24, 돌리기 전에 커밋):
  C1 표면각 90% 중앙: 우리 < 90° (계단이 사라졌다 · 09-24 exe 는 35장 모두 90.0°)
  C2 F@2mm 겉 35장 중앙 >= 0.760 (0.790 - 0.03 안쪽 — 떨어져도 조금)
  C3 모든 장 닫힌 메시 · 같은 줄 두 번 -> 같은 해시 (3장 뽑아 확인)
"""
import json
import os
import sys
import time

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
import 지표 as J             # noqa: E402
import 그림 as G             # noqa: E402
import dsl as D              # noqa: E402
from exe벤치 import 시트    # noqa: E402
from 넓은시트 import 잘린, 넓게  # noqa: E402

OUT = os.path.join(A.OUT, "곡선")


def 한장(c):
    뿌리 = 넓게 if c in 잘린 else 시트
    앞, 옆 = (G.마스크(os.path.join(뿌리, c, n + ".png")) for n in ("front", "side"))
    t = time.time()
    후보, 동점 = G.고르기(앞, 옆, 150.0)
    V, F = D.실행(후보[0]["줄"])
    d = os.path.join(OUT, c)
    os.makedirs(d, exist_ok=True)
    trimesh.Trimesh(V, F, process=False).export(os.path.join(d, "메시.stl"))
    json.dump({"해시": D.해시(V, F), "후보": [{k: h[k] for k in ("이름", "점수", "줄")} for h in 후보], "동점": 동점},
              open(os.path.join(d, "결과.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return time.time() - t


def 재기():
    기록 = {}
    for c in json.load(open(os.path.join(A.OUT, "겉지표.json"), encoding="utf-8")):
        p = os.path.join(OUT, c, "메시.stl")
        if not os.path.exists(p):
            continue
        m = trimesh.load(p, force="mesh")
        g = A.정답(c)
        우 = J.mm메시(A.창(m.vertices), m.faces)
        f, 정, 재 = J.F겉(우, J.겉점(g)[0])
        r, _ = J.재기(우, J.mm메시(g["메시"].vertices, g["메시"].faces))
        기록[c] = {"F@2mm 겉": f, "챔퍼 중앙": r["챔퍼 중앙"], "새 각도 중앙": r["새 각도 중앙"], "표면각 우리": r["표면각 우리"],
                  "표면각 정답": r["표면각 정답"], "닫힘": bool(m.is_watertight),
                  "1순위": json.load(open(os.path.join(OUT, c, "결과.json"), encoding="utf-8"))["후보"][0]["이름"]}
    return 기록


def main(n):
    os.makedirs(OUT, exist_ok=True)
    남은 = [c for c in json.load(open(os.path.join(A.OUT, "겉지표.json"), encoding="utf-8"))
          if not os.path.exists(os.path.join(OUT, c, "메시.stl"))]
    for c in 남은[:n]:
        print("  %-32s %.0f초" % (c[:32], 한장(c)), flush=True)
    print("남은 %d장" % (len(남은) - min(n, len(남은))))
    if len(남은) <= n:
        기록 = 재기()
        json.dump(기록, open(os.path.join(A.OUT, "곡선.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        med = lambda f: float(np.median([f(v) for v in 기록.values()]))
        print("F@2mm 겉 %.3f · 챔퍼 %.2f · 새 각도 %.3f · 표면각 90%% 우리 %.1f° 정답 %.1f° · 닫힘 %d/%d" % (
            med(lambda v: v["F@2mm 겉"]), med(lambda v: v["챔퍼 중앙"]), med(lambda v: v["새 각도 중앙"]),
            med(lambda v: v["표면각 우리"][1]), med(lambda v: v["표면각 정답"][1]), sum(v["닫힘"] for v in 기록.values()), len(기록)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
