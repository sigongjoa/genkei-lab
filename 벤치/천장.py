"""그림 두 장의 천장 — 층 타원(C)이 어디까지 갈 수 있나. 한 번에 한 가지씩만 다르게 세 판을 잰다.

    python 벤치/천장.py        -> out/천장.json

  H  두 장 visual hull   앞 · 옆 실루엣을 각각 밀어내 겹친 곳(켜마다 네모 단면). 두 그림과 어긋나지 않는 가장 큰 모양
  C  층 타원             앱의 층타원 그대로 (켜마다 앞 구간 하나에 타원, 깊이 · y = 옆 그림 한 줄 전체)
  Cd C + 참 깊이         C 와 같은데 구간마다 깊이 · y 만 **정답 메시 단면**에서 가져온다 — 옆 그림 겹침(외부 평가 ㄷ)의 값

H 는 IoU 의 상한이 아니다 — 팔다리는 둥글어 네모 단면이 넘칠 수 있다. Cd 는 정답을 보니 앱에서는 못 쓴다(천장 재기용).
입력은 시트 그림 두 장 + 키 150 mm, 채점은 exe 벤치와 같은 자(어댑터 IoU3D).

예측 (09-22, 돌리기 전에 커밋):
  T1 사람형 35장 중앙 H < C — 팔다리가 둥글어 네모 단면이 진다
  T2 사람형 35장 중앙 Cd − C >= 0.05 — 옆 그림 겹침이 적어도 0.05 를 먹는다
  T3 사람형 35장 중앙 Cd <= 0.85 — 깊이를 맞혀도 타원 단면 · 켜 60 의 한계가 남는다
  T4 도형: 상자 H >= 0.99 (H 가 제대로 지어졌나 확인)

돌린 뒤 (09-22, 46장) — 넷 다 맞음:
  사람형 35장 중앙  H 0.662 < C 0.745 < Cd 0.802   (마네킹 0.801 · 0.870 · 0.914 / 실물 0.654 · 0.722 · 0.764)
  → 옆 그림 겹침(깊이)이 먹는 몫은 중앙 +0.057. 큰 곳: 검들기 +0.28 · 팔앞뒤 +0.23 · knight 둘 +0.15 · +0.17 · monster +0.11.
  → 깊이를 정답으로 줘도 0.80 — 이 계열(켜 60 · 타원 단면)의 천장. 남은 몫은 단면 꼴 · 사람 아닌 물건.
  단면 꼴: 켜마다 네모(H)가 이기는 10장(상자 · 메카 · 로봇 · 모자 · 조각상 · 검들기 · 마녀 · 마법사). 장마다 H · C 중 나은 것을 고르면 0.767
    — 그런데 H 와 C 는 앞 · 옆 실루엣이 같아 앱은 못 가른다(도형 동점 넷과 같은 까닭). 범주 기본값이나 사람 고르기 몫.
"""
import json
import os
import sys

import numpy as np
import trimesh
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
import 그림 as G            # noqa: E402
import dsl as D             # noqa: E402
import exe벤치 as X          # noqa: E402

키, 칸 = 150.0, 0.5           # mm · H 격자 한 칸


def 마스크들(case):
    d = os.path.join(A.원화3d, "out", "시트", case)
    return G.마스크(os.path.join(d, "front.png")), G.마스크(os.path.join(d, "side.png"))


def hull(앞, 옆):
    """앞 · 옆 실루엣을 밀어내 겹친 모양 -> (V, F) mm. 조각 좌표: x = 앞 가로, y = -(옆 가로), z = 높이."""
    fa, fo = G.틀(앞, 키), G.틀(옆, 키)
    x0, x1 = fa.범위()[:2]
    u0, u1 = fo.범위()[:2]
    xs, us, zs = np.arange(x0, x1, 칸) + 칸 / 2, np.arange(u0, u1, 칸) + 칸 / 2, np.arange(0, 키, 칸) + 칸 / 2
    fx = np.clip((fa.c + xs / fa.s).astype(int), 0, 앞.shape[1] - 1)
    fu = np.clip((fo.c + us / fo.s).astype(int), 0, 옆.shape[1] - 1)
    rz_a = np.clip((fa.bot - zs / fa.s).astype(int), 0, 앞.shape[0] - 1)
    rz_o = np.clip((fo.bot - zs / fo.s).astype(int), 0, 옆.shape[0] - 1)
    Fr = 앞[rz_a][:, fx]                       # (z, x)
    Sd = 옆[rz_o][:, fu]                       # (z, u)
    vol = Fr[:, :, None] & Sd[:, None, :]      # (z, x, u)
    vol = np.pad(vol, 1)
    V, F, _, _ = measure.marching_cubes(vol.astype(np.float32), 0.5)
    z, x, u = (V[:, 0] - 1) * 칸, (V[:, 1] - 1) * 칸 + x0, (V[:, 2] - 1) * 칸 + u0
    return np.stack([x, -u, z], 1), F[:, ::-1]


def 참단면(답mm, z, x0, x1):
    """정답 메시(조각 좌표 mm)의 높이 z 단면에서 x 가 [x0, x1] 인 점들의 (y 가운데, 깊이). 없으면 None."""
    s = 답mm.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if s is None:
        return None
    P = np.asarray(s.vertices)
    P = P[(P[:, 0] >= x0) & (P[:, 0] <= x1)]
    if len(P) < 2:
        return None
    return (P[:, 1].min() + P[:, 1].max()) / 2, P[:, 1].max() - P[:, 1].min()


def 층타원_참깊이(앞, 옆, 답mm, 층수=60):
    """그림.층타원 과 같은 켜 · 같은 앞 구간, 깊이 · y 만 정답 단면에서."""
    fa, fo = G.틀(앞, 키), G.틀(옆, 키)
    rows = np.nonzero(앞.any(1))[0]
    경계 = np.linspace(rows.min(), rows.max() + 1, 층수 + 1)
    토막 = []
    for r0, r1 in zip(경계[:-1], 경계[1:]):
        r = int((r0 + r1) / 2)
        z0, z1 = (fa.bot - r1) * fa.s, (fa.bot - r0) * fa.s + 0.01
        기본 = G._옆줄(옆, fo, (z0 + z1) / 2)
        for a, b in G._줄조각(앞[r]):
            xa, xb = (a - fa.c) * fa.s, (b - fa.c) * fa.s
            참 = 참단면(답mm, (z0 + z1) / 2, xa, xb) or 기본
            if not 참:
                continue
            y, 깊이 = 참
            x, 너비 = (xa + xb) / 2, xb - xa
            토막.append("로프트(점=[[%.1f, %.1f, %.2f], [%.1f, %.1f, %.2f]], w=[%.1f, %.1f], d=[%.1f, %.1f])"
                      % (x, y, z0, x, y, z1, 너비, 너비, max(깊이, 0.5), max(깊이, 0.5)))
    return " + ".join(토막)


def main():
    기록 = {}
    for g in ("도형", "마네킹", "실물"):
        for c in X.묶음[g]:
            앞, 옆 = 마스크들(c)
            답 = A.정답(c)
            답mm = trimesh.Trimesh(A.조각좌표(답["메시"].vertices) + [0, 0, 키 / 2], 답["메시"].faces, process=False)
            r = {"묶음": g}
            V, F = hull(앞, 옆)
            r["H"] = A.채점(V, F, 답)["IoU3D"]
            r["C"] = A.채점(*D.실행(G.층타원(앞, 옆, 키)), 답)["IoU3D"]
            r["Cd"] = A.채점(*D.실행(층타원_참깊이(앞, 옆, 답mm)), 답)["IoU3D"]
            기록[c] = r
            print("  %-4s %-36s H %.3f · C %.3f · Cd %.3f" % (g, c[:36], r["H"], r["C"], r["Cd"]), flush=True)
    사람 = [v for v in 기록.values() if v["묶음"] != "도형"]
    med = lambda k, xs=사람: float(np.median([v[k] for v in xs]))
    h, c, cd = med("H"), med("C"), med("Cd")
    판정 = [("T1 사람형 H < C", h < c, "H %.3f · C %.3f" % (h, c)),
            ("T2 사람형 Cd − C >= 0.05", cd - c >= 0.05, "Cd %.3f − C %.3f = %+.3f" % (cd, c, cd - c)),
            ("T3 사람형 Cd <= 0.85", cd <= 0.85, "Cd %.3f" % cd),
            ("T4 도형 상자 H >= 0.99", 기록["도형_상자"]["H"] >= 0.99, "상자 H %.3f" % 기록["도형_상자"]["H"])]
    print("\n예측")
    for 말, 맞, 글 in 판정:
        print("  %s %s — %s" % ("○" if 맞 else "✗", 말, 글))
    for g in ("마네킹", "실물"):
        xs = [v for v in 사람 if v["묶음"] == g]
        print("  %s %d장 중앙 — H %.3f · C %.3f · Cd %.3f" % (g, len(xs), med("H", xs), med("C", xs), med("Cd", xs)))
    json.dump({"기록": 기록, "예측": [{"예측": a, "맞음": bool(b), "값": t} for a, b, t in 판정]},
              open(os.path.join(A.OUT, "천장.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
