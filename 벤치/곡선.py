"""다루는 건 전부 곡선 (09-24 사용자: 「아예 다루는 게 곡선이어야」 · 「F 가 떨어져도 형태가 비슷하면 된다」).

    python 벤치/곡선.py [n] [이름]     앞에서부터 n 장 더(이어서) -> out/<이름>/<케이스>/{메시.stl, 결과.json} · out/<이름>.json
                                     이름 = 곡선(09-24 줄기 + 부드럽게) · 켜부피(09-24 줄기 없음)

  후보에서 계단(혼합 · 층 타원)을 빼고 매끈한 판(혼합 겹침 · 층 겹침 · 부위 둘)을 dsl 「부드럽게」로 감쌌다:
  조각마다 부호 거리장 -> 다항식 부드러운 최솟값(반경 3 mm) -> manifold level_set. 이음매(겨드랑이 · 가랑이 · 목)가 둥글다.
  exe 대신 파이썬 고르기(그림.고르기 — exe 와 같은 함수)로 35장. 잘렸던 4장은 넓은 시트. 판단은 갤러리 그림으로.

예측 (09-24, 돌리기 전에 커밋):
  C1 표면각 90% 중앙: 우리 < 90° (계단이 사라졌다 · 09-24 exe 는 35장 모두 90.0°)
  C2 F@2mm 겉 35장 중앙 >= 0.760 (0.790 - 0.03 안쪽 — 떨어져도 조금)
  C3 모든 장 닫힌 메시 · 같은 줄 두 번 -> 같은 해시 (3장 뽑아 확인)

돌린 뒤 (09-24) — 셋 다 맞음(C3 은 고친 뒤):
  F@2mm 겉 35장 중앙 0.790(계단 exe) -> 0.783 · 챔퍼 0.87 -> 0.89 mm · 새 각도 0.886 -> 0.889 ·
  표면각 90% 90.0° -> 12.7° (계단이 없다. 정답 83.5° 는 게임 메시의 각진 모서리).
  고친 둘 — 그림을 보고: ① 칠한 칸의 거리는 1 mm 격자 계단을 탔다 -> 거리장 가우스 흐림 σ 1 칸
  ② 면이 한 점에서 맞닿는 곳의 겹친 정점을 STL 이 합쳐 32/35 만 닫힘 -> 0.002 mm 안 정점을 0.01 mm 떼어 35/35.
  고르기는 흐림 전 후보로 했다(실루엣 기준이라 흐림이 거의 안 바꾼다) — 메시만 흐림 뒤 코드로 다시 뽑았다.
  0.03 넘게 떨어진 장: hairsample_male 0.888 -> 0.795 · 마법사 셋(0.02~0.05). 한 장 1~9 분(큰 물체가 느리다 — 파이썬 level_set 콜백).
  그림: out/곡선확대_*.png (정답 | 계단 | 곡선, 면 그늘) · out/갤러리곡선_*.png

켜부피 (09-24 사용자 「슬라이스」 -> 「고쳐서 제대로」): 후보 = 부위 둘(부드럽게) · 혼합 켜부피(켜 부피 몸 + 팔 로프트, 부드럽게) · 켜 부피.
  켜 부피 = dsl 켜부피 — 높이 1 mm 마다 (앞 구간 × 옆 구간) 타원을 칸에 칠하고 z 로도 흐림. 줄기 · 뚜껑 원반 없음.
  곡면 = 마칭 큐브 먼저(빠름), 안 되면 level_set. 예측 (돌리기 전에 커밋) — `python 벤치/곡선.py 35 켜부피`:
  K1 F@2mm 겉 35장 중앙 >= 0.770 (곡선 0.783)
  K2 닫힘 35/35 · 같은 줄 두 번 같은 해시
  K3 한 장 평균 < 30 초 (곡선은 1~9 분)
  돌린 뒤 — 셋 다 맞음: F@2mm 겉 0.791 (곡선 0.783 · 계단 exe 0.790) · 챔퍼 0.90 · 새 각도 0.898 · 표면각 90% 9.6° ·
  닫힘 35/35 · 해시 같음(3장) · 35장 10 분 안(장당 ~15 초). 1순위 켜 부피 33 · 혼합 켜부피 2.
  곡선 대비 0.03 넘게: 오름 avatarsample_g 0.774 -> 0.831 · 상자 0.282 -> 0.312 / 내림 치마 마네킹 0.874 -> 0.843 ·
  avatarsample_e 0.901 -> 0.858 · hairsample_female 0.832 -> 0.791. 그림 out/켜부피확대_*.png · out/갤러리켜부피_*.png:
  팔 밑 판 · 이음매 턱 · 끊긴 원반이 사라졌다. 남은 것: 기운 모자 챙(마녀 · 노움 — 두 장 한계) · 팔이 몸통에 붙은 곳의 날개(겹침 애매).
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

OUT = os.path.join(A.OUT, sys.argv[2] if len(sys.argv) > 2 else "곡선")
이름 = os.path.basename(OUT)


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
        json.dump(기록, open(os.path.join(A.OUT, 이름 + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        med = lambda f: float(np.median([f(v) for v in 기록.values()]))
        print("F@2mm 겉 %.3f · 챔퍼 %.2f · 새 각도 %.3f · 표면각 90%% 우리 %.1f° 정답 %.1f° · 닫힘 %d/%d" % (
            med(lambda v: v["F@2mm 겉"]), med(lambda v: v["챔퍼 중앙"]), med(lambda v: v["새 각도 중앙"]),
            med(lambda v: v["표면각 우리"][1]), med(lambda v: v["표면각 정답"][1]), sum(v["닫힘"] for v in 기록.values()), len(기록)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
