"""부품 = 도색 단위 (09-24, 원형사 교본 4장 규칙 1 · 2 — 「색이 다르면 부품이 다르다 · 피부는 한 부품」).

    python 벤치/도색단위.py      -> out/도색단위.json · out/도색단위_*.png

  정답 — VRoid 11장의 원본(_model.glb) 재질을 도색 단위로 묶는다:
      피부 = Body · Face · FACE · EYE(눈 · 입은 피부 위에 그린다) · 머리카락 = HAIR · 상의 Tops · 하의 Bottoms · 신발 Shoes · 장식 Accessory.
      원본을 (x, -z, y) 로 돌리고 win.normalize 하면 벤치 정답(truth.glb)과 정확히 겹친다(중앙 거리 0).
  우리 — 켜부피 메시(out/켜부피) 겉면 점마다 **보이는 그림**(정면 · 뒤 · 옆, 점 z 버퍼)에서 색을 읽고,
      Lab 색을 k-평균(k=8, 씨 고정)으로 묶어 작은 묶음(< 2 %)은 가까운 색에 합친다. 안 보이는 점은 가까운 점의 묶음.
  자 — 정답 단위는 점마다 가장 가까운 정답 점의 단위.
      라벨 정확도 = 묶음마다 다수 단위로 옮겼을 때 맞는 몫(넓이 가중).
      부품 순도 = 부품마다 다수 단위가 차지하는 몫의 넓이 가중 평균. 지금 키트 부품(사람형 규칙 6 개, out/exe/<케이스>/부품)과
      색 부품(같은 묶음의 이어진 겉면 조각)을 같은 자로.

예측 (09-24, 돌리기 전에 커밋) — VRoid 11장:
  D1 색 묶음 라벨 정확도 중앙 >= 0.85
  D2 지금 키트 부품(규칙 6 개) 순도 중앙 <= 0.75 (몸통 = 상의 + 하의 + 피부가 섞인다)
  D3 색 부품 순도 중앙 >= 0.90
"""
import json
import os
import re
import sys

import numpy as np
import trimesh
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import 어댑터 as A           # noqa: E402
import mask2d                # noqa: E402  (원화3d win 이 길을 잡아 둔다)
import win as W              # noqa: E402

시트 = os.path.join(A.원화3d, "out", "시트")
단위들 = ["피부", "머리카락", "상의", "하의", "신발", "장식"]
VRoid = ["avatarsample_d", "avatarsample_d_darkness", "avatarsample_e", "avatarsample_f", "avatarsample_g", "base_female", "base_male",
         "hairsample_female", "hairsample_male", "sakurada_fumiriya", "sendagaya_shino"]


def 단위(재질):
    if re.search(r"HAIR", 재질):
        return "머리카락"
    for k, u in (("Tops", "상의"), ("Bottoms", "하의"), ("Shoes", "신발"), ("Accessory", "장식")):
        if k in 재질:
            return u
    return "피부"                                                          # SKIN · FACE · EYE


def 정답점(case, n=120000):
    s = trimesh.load(os.path.join(시트, case, "_model.glb"))
    P, L, F = [], [], []
    for m in s.dump():
        V = np.asarray(m.vertices)[:, [0, 2, 1]] * [1, -1, 1]
        P.append(V)
        L.append((단위(m.visual.material.name), len(V)))
    전 = np.vstack(P)
    lo, hi = 전.min(0), 전.max(0)
    s_ = 1.0 / (hi[2] - lo[2])
    중 = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]) * s_
    점, 라 = [], []
    총넓이 = sum(m.area for m in s.dump())
    for m in s.dump():
        k = max(1, int(n * m.area / 총넓이))
        q = trimesh.sample.sample_surface(m, k, seed=0)[0][:, [0, 2, 1]] * [1, -1, 1] * s_ - 중
        점.append(q)
        라 += [단위(m.visual.material.name)] * len(q)
    return np.vstack(점), np.array(라)


def 창틀(V):
    """조각 좌표(mm) -> 창 좌표 변환을 **전체 메시**로 정한다(부품에 같은 변환)."""
    V = np.asarray(V) * A.뒤집기
    lo, hi = V.min(0), V.max(0)
    s = 1.0 / (hi[2] - lo[2])
    중 = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]) * s
    return lambda X: np.asarray(X) * A.뒤집기 * s - 중


def _lab(rgb):
    from skimage.color import rgb2lab
    return rgb2lab(rgb[None].astype(float) / 255.0)[0]


def 색라벨(case, 창V, 창F, n=40000, k=8):
    """우리 겉면 점 · 법선 · 보이는 그림의 색 -> 묶음 번호."""
    m = trimesh.Trimesh(창V, 창F, process=False)
    P, fi = trimesh.sample.sample_surface(m, n, seed=1)
    N = m.face_normals[fi]
    그림 = {v: np.asarray(Image.open(os.path.join(시트, case, v + ".png")).convert("RGBA")) for v in ("front", "back", "side")}
    색 = np.full((len(P), 3), np.nan)
    for v, (축, 부호), 깊축, 깊부호, 보임 in (("front", (0, 1), 1, 1, N[:, 1] > 0.25), ("back", (0, -1), 1, -1, N[:, 1] < -0.25),
                                          ("side", (1, 1), 0, 1, np.abs(N[:, 0]) > 0.5)):
        py, px = mask2d.topix(부호 * P[:, 축], P[:, 2])
        깊 = 깊부호 * P[:, 깊축] if v != "side" else np.abs(P[:, 0])
        z = np.full((mask2d.RES, mask2d.RES), -np.inf)
        np.maximum.at(z, (py, px), 깊)
        from scipy.ndimage import maximum_filter
        z = maximum_filter(z, 3)
        앞 = 보임 & (깊 >= z[py, px] - 0.012) & np.isnan(색[:, 0])
        img = 그림[v]
        a = img[py, px, 3] > 127
        ok = 앞 & a
        색[ok] = img[py[ok], px[ok], :3]
    있 = ~np.isnan(색[:, 0])
    L = _lab(색[있])
    rng = np.random.default_rng(0)
    cen, lab = kmeans2(L, k, seed=rng, minit="++")
    몫 = np.bincount(lab, minlength=k) / len(lab)
    작 = np.nonzero(몫 < 0.02)[0]
    큰 = np.nonzero(몫 >= 0.02)[0]
    for s_ in 작:
        lab[lab == s_] = 큰[np.argmin(np.linalg.norm(cen[큰] - cen[s_], axis=1))]
    전 = np.full(len(P), -1)
    전[있] = lab
    _, j = cKDTree(P[있]).query(P[~있])
    전[~있] = lab[j]
    return P, fi, 전


def 순도(단위배열, 무리):
    """무리(부품 · 묶음)마다 다수 단위 몫의 가중 평균 · 무리 -> 다수 단위."""
    tot, 맞, 대응 = 0, 0, {}
    for g in np.unique(무리):
        u = 단위배열[무리 == g]
        vals, cnt = np.unique(u, return_counts=True)
        대응[g] = vals[np.argmax(cnt)]
        tot += len(u)
        맞 += cnt.max()
    return 맞 / max(tot, 1), 대응


def 색부품(m, fi, 라벨):
    """같은 묶음 라벨의 이어진 면 조각 = 색 부품. 면 라벨 = 그 면에 떨어진 점들의 다수."""
    nf = len(m.faces)
    면라 = np.full(nf, -1)
    for f, l in zip(fi, 라벨):
        면라[f] = l                                                       # 대충 — 점이 여럿이면 마지막
    adj = m.face_adjacency
    빈 = 면라 < 0
    while 빈.any():                                                       # 점 안 떨어진 면은 이웃에서 채움
        a, b = adj[:, 0], adj[:, 1]
        for x, y in ((a, b), (b, a)):
            t = 빈[x] & ~빈[y]
            면라[x[t]] = 면라[y[t]]
        새빈 = 면라 < 0
        if 새빈.sum() == 빈.sum():
            break
        빈 = 새빈
    같 = 면라[adj[:, 0]] == 면라[adj[:, 1]]
    g = coo_matrix((np.ones(같.sum()), (adj[같, 0], adj[같, 1])), shape=(nf, nf))
    _, comp = connected_components(g, directed=False)
    return comp[fi]


def 한장(c):
    답P, 답L = 정답점(c)
    T = cKDTree(답P)
    곡 = trimesh.load(os.path.join(A.OUT, "켜부피", c, "메시.stl"), force="mesh")
    틀 = 창틀(곡.vertices)
    V, F = 틀(곡.vertices), np.asarray(곡.faces)
    P, fi, 라 = 색라벨(c, V, F)
    참 = 답L[T.query(P)[1]]
    정확, 대응 = 순도(참, 라)
    m = trimesh.Trimesh(V, F, process=False)
    조각 = 색부품(m, fi, 라)
    큰조각 = np.bincount(조각)
    부품수 = int((큰조각 >= 0.01 * len(조각)).sum())
    색순도, _ = 순도(참, 조각)
    # 지금 키트 부품 (exe 메시와 같은 틀)
    전 = trimesh.load(os.path.join(A.OUT, "exe", c, "메시.stl"), force="mesh")
    틀2 = 창틀(전.vertices)
    Q, 무 = [], []
    for 이름 in sorted(os.listdir(os.path.join(A.OUT, "exe", c, "부품"))):
        p = trimesh.load(os.path.join(A.OUT, "exe", c, "부품", 이름), force="mesh")
        q = trimesh.sample.sample_surface(trimesh.Trimesh(틀2(p.vertices), p.faces, process=False), 4000, seed=2)[0]
        Q.append(q)
        무 += [이름[:-4]] * len(q)
    Q = np.vstack(Q)
    규칙순도, 규칙대응 = 순도(답L[T.query(Q)[1]], np.array(무))
    return {"라벨 정확도": round(float(정확), 3), "색 부품 순도": round(float(색순도), 3), "색 부품 수(1 % 넘는)": 부품수,
            "규칙 부품 순도": round(float(규칙순도), 3), "규칙 부품 다수 단위": 규칙대응,
            "묶음 -> 단위": {int(k): v for k, v in 대응.items()},
            "정답 단위 몫": {u: round(float((참 == u).mean()), 3) for u in 단위들 if (참 == u).any()}}, (V, F, P, 라, 참)


def main():
    기록, 그림들 = {}, {}
    for c in VRoid:
        r, g = 한장(c)
        기록[c] = r
        그림들[c] = g
        print("%-26s 라벨 %.3f · 색 부품 순도 %.3f (%d 개) · 규칙 부품 순도 %.3f" % (
            c, r["라벨 정확도"], r["색 부품 순도"], r["색 부품 수(1 % 넘는)"], r["규칙 부품 순도"]), flush=True)
    med = lambda k: float(np.median([v[k] for v in 기록.values()]))
    print("중앙 — 라벨 %.3f · 색 부품 순도 %.3f · 규칙 부품 순도 %.3f" % (med("라벨 정확도"), med("색 부품 순도"), med("규칙 부품 순도")))
    for 말, ok in (("D1 라벨 >= 0.85", med("라벨 정확도") >= 0.85), ("D2 규칙 순도 <= 0.75", med("규칙 부품 순도") <= 0.75),
                   ("D3 색 부품 순도 >= 0.90", med("색 부품 순도") >= 0.90)):
        print("  %s %s" % ("○" if ok else "✗", 말))
    json.dump(기록, open(os.path.join(A.OUT, "도색단위.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 그림들


if __name__ == "__main__":
    main()
