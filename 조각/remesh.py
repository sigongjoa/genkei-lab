# -*- coding: utf-8 -*-
"""**리메시** — 메시를 고른 간격으로 다시 깐다 (2026-09-22).

    python remesh.py 검사      구 · 성긴 부품 · 오메가몬 형태로 재 본다 (결정성 · 모서리 고름 · 모양 유지)

사용자(09-22): 「이거 기능 만들어줘봐봐」 — 조각 앱 다음 판 ① 리메시.
붓은 정점만 옮긴다. 정점이 성기거나 크기가 들쭉날쭉하면 붓이 각지게 먹는다(예제 `오메가몬_부품00_성김.ply`:
모서리 중앙 3.5 mm · 최대 11 mm). 그래서 붓질 전에 고르게 다시 깐다.

    ① 표면을 격자에 찍는다     모서리가 간격/2 보다 길면 쪼갠 뒤(trimesh subdivide_to_size) 정점을 칸에 떨군다
    ② 속을 채운다              껍데기를 한 칸 불려 틈을 막고 binary_fill_holes · 다시 한 칸 깎는다
    ③ 거리장                   안쪽 거리 − 바깥 거리 (EDT) — 0/1 격자 대신 연속 값이라 계단이 줄어든다
    ④ 마칭 큐브                거리장 **1** 면 (skimage) — 0 에서 뜨면 반 칸 바깥으로 밀린다(09-22 구에서 0.53 mm 잼)
    ⑤ 다듬기                   Taubin 5 회 — 부피를 덜 먹는 매끈 (lamb 0.5 · nu −0.53)

같은 (V, F, 간격)이면 같은 결과가 나온다 — 난수를 안 쓴다. 타임라인에서 명령 한 칸이 된다.
ponytail: 파이썬 · 격자 전체 — 150 mm / 0.5 mm 면 300³. 느려지면 C++ 로 (붓 엔진처럼).
"""
import sys
import time

import numpy as np
from scipy import ndimage


def 리메시(V, F, 간격=0.8, 다듬기=5):
    import trimesh
    from skimage.measure import marching_cubes
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    v, f = trimesh.remesh.subdivide_to_size(V, F, max_edge=간격 / 2, max_iter=12)
    lo = v.min(0) - 3 * 간격
    ijk = np.floor((v - lo) / 간격).astype(np.int64)
    shape = tuple(ijk.max(0) + 4)
    껍 = np.zeros(shape, bool)
    껍[tuple(ijk.T)] = True
    막 = ndimage.binary_dilation(껍, iterations=1)
    속 = ndimage.binary_fill_holes(막)
    속 = ndimage.binary_erosion(속, iterations=1) | 껍
    거리 = ndimage.distance_transform_edt(속) - ndimage.distance_transform_edt(~속)
    거리 = ndimage.gaussian_filter(거리.astype(np.float64), 0.6)
    # 껍데기 칸은 속으로 쳤다 -> 껍데기 칸 중심(= 표면이 지나간 자리)의 거리 값이 1 이다. 0 이 아니라 1 에서 뜬다
    nv, nf, _, _ = marching_cubes(거리, level=1.0, spacing=(간격,) * 3)
    nv = nv + lo + 간격 / 2
    m = trimesh.Trimesh(nv, nf[:, ::-1], process=True)          # skimage 감김을 바깥 법선으로
    if 다듬기:
        trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=다듬기)
    if m.volume < 0:
        m.invert()
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def 모서리(V, F):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    L = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    return float(np.median(L)), float(np.percentile(L, 99)), float(L.max())


def 검사():
    import os
    import trimesh
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import engine as E
    예 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "예제")
    대상 = [("구 (세분 3)", *E.기본메시(세분=3))]
    for 이름 in ("오메가몬_부품00_성김.ply", "오메가몬_형태.ply"):
        p = os.path.join(예, 이름)
        if os.path.exists(p):
            m = trimesh.load(p, force="mesh", process=True)
            V = np.asarray(m.vertices); V = (V - (V.max(0) + V.min(0)) / 2) * (150 / np.ptp(V[:, 2]))
            대상.append((이름, V, np.asarray(m.faces)))
    for 이름, V, F in 대상:
        t = time.time()
        간격 = 0.8
        a = 리메시(V, F, 간격)
        초 = time.time() - t
        b = 리메시(V, F, 간격)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), "결정성"
        m0 = trimesh.Trimesh(V, F, process=False)
        m1 = trimesh.Trimesh(*a, process=False)
        # 모양 유지: 새 정점이 옛 표면에서 얼마나 떨어졌나
        _, d, _ = trimesh.proximity.closest_point(m0, a[0][:: max(1, len(a[0]) // 3000)])
        e0, e1 = 모서리(V, F), 모서리(*a)
        print("%-24s 정점 %7d -> %7d · 모서리 중앙 %.2f -> %.2f mm (99%% %.2f -> %.2f) · 표면 거리 중앙 %.3f mm 최대 %.2f · 닫힘 %s · %.1f초"
              % (이름, len(V), len(a[0]), e0[0], e1[0], e0[1], e1[1], np.median(d), d.max(), m1.is_watertight, 초))
        assert e1[1] < 2.5 * 간격, "모서리가 고르다 (99% 가 간격의 2.5 배 안)"
        assert np.median(d) < 0.5 * 간격, "모양 유지 (표면 거리 중앙 < 간격 절반)"
    print("검사 통과 — 결정성 · 모서리 고름 · 모양 유지")


if __name__ == "__main__":
    검사()
