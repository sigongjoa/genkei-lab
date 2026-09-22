# -*- coding: utf-8 -*-
"""**조각 엔진 C++ 판을 파이썬에서 부른다** — engine.조각 과 같은 모양 (2026-09-22).

    python engine_cpp.py 검사     파이썬 판(정답기)과 붓 다섯 · 대칭 · 반전을 같은 붓질로 대 본다 + 속도

사용자(09-22): 「파이썬이라서 느린 거야? 그러면 C++ 로 짜 봐」 — 재 보니 느린 몫의 대부분은 파이썬이 아니라
**자국마다 메시 전체를 다시 계산하는 방식**이었다(21만 정점 · 자국당 162 ms 중 전체 법선 102 · 공간 색인 42).
C++ 판(`cpp/sculpt.cpp`)은 붓이 닿은 곳만 계산한다. 공식 · 간격 · 대칭 · 잡기는 engine.py 와 줄마다 같다.

**같음의 뜻** — 두 엔진은 덧셈 순서가 달라 마지막 자리가 다를 수 있다. 그래서 파이썬 판과는 **정점 최대 차이**로 대고
(문턱 1e-6 mm), 같은 C++ 판끼리의 재생은 해시로 댄다.
"""
import ctypes
import hashlib
import os
import sys

import numpy as np

import engine as E

HERE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
_dll = ctypes.CDLL(os.path.join(HERE, "cpp", "sculpt.dll"))
_dll.sc_create.restype = ctypes.c_void_p
_dll.sc_create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
for nm, args in [("sc_destroy", [ctypes.c_void_p]), ("sc_reset", [ctypes.c_void_p]),
                 ("sc_stroke_begin", [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]),
                 ("sc_get_V", [ctypes.c_void_p, ctypes.c_void_p]),
                 ("sc_get_some", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p])]:
    getattr(_dll, nm).argtypes = args
_dll.sc_add_point.restype = ctypes.c_int
_dll.sc_add_point.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_void_p, ctypes.c_int]
_dll.sc_dab_count.restype = ctypes.c_int
_dll.sc_dab_count.argtypes = [ctypes.c_void_p]


class 조각C:
    """engine.조각 과 같은 이름 · 같은 저널. 붓 계산만 C++."""

    def __init__(self, V, F):
        self.V0 = np.ascontiguousarray(V, np.float64)
        self.F = np.ascontiguousarray(F, np.int32)
        self._h = _dll.sc_create(self.V0.ctypes.data, len(self.V0), self.F.ctypes.data, len(self.F))
        self._buf = np.zeros(len(self.V0), np.int32)
        self.저널 = []
        self._획 = None

    def __del__(self):
        if getattr(self, "_h", None):
            _dll.sc_destroy(self._h)
            self._h = None

    @property
    def V(self):
        out = np.empty((len(self.V0), 3), np.float64)
        _dll.sc_get_V(self._h, out.ctypes.data)
        return out

    def 위치(self, idx):
        idx = np.ascontiguousarray(idx, np.int32)
        out = np.empty((len(idx), 3), np.float64)
        if len(idx):
            _dll.sc_get_some(self._h, idx.ctypes.data, len(idx), out.ctypes.data)
        return out

    def 해시(self):
        return hashlib.sha1(np.round(self.V, 3).tobytes()).hexdigest()[:12]

    def 획_시작(self, 붓, 반지름, 세기, 대칭=False, 반전=False):
        assert 붓 in E.붓들, 붓
        self._획 = {"붓": 붓, "반지름": float(반지름), "세기": float(세기), "대칭": bool(대칭), "반전": bool(반전), "점": [], "자국수": 0}
        _dll.sc_stroke_begin(self._h, E.붓들.index(붓), float(반지름), float(세기), int(대칭), int(반전))
        return self._획

    def 점_더하기(self, p):
        q = [round(float(x), 4) for x in p]                     # 저널에 남는 값 그대로 계산한다 (engine.py 와 같다)
        self._획["점"].append(q)
        k = _dll.sc_add_point(self._h, q[0], q[1], q[2], self._buf.ctypes.data, len(self._buf))
        return self._buf[:k].tolist()

    def 획_끝(self):
        s = self._획
        self._획 = None
        if s is None or not s["점"]:
            return None
        s["자국수"] = _dll.sc_dab_count(self._h)
        s["해시"] = self.해시()
        self.저널.append(s)
        return s

    def 처음으로(self):
        _dll.sc_reset(self._h)


def 재생(저널, V, F):
    e = 조각C(V, F)
    for 획 in 저널:
        E.재생한획(e, 획)
    return e


def 검사():
    import time
    V, F = E.기본메시(세분=5)
    선 = [[x, -49.0, 5.0 + 3 * np.sin(x / 7)] for x in np.linspace(-25, 25, 23)]
    붓질 = [("그리기", 9, 0.6, False, False), ("부풀리기", 12, 0.4, True, False), ("매끈", 14, 0.5, True, False),
           ("주름", 6, 0.7, False, False), ("그리기", 7, 0.5, True, True), ("주름", 5, 0.5, True, True)]
    py, cc = E.조각(V, F), 조각C(V, F)
    최대 = []
    for 붓, R, I, 대, 반 in 붓질:
        for e in (py, cc):
            e.획_시작(붓, R, I, 대, 반)
            for p in 선:
                e.점_더하기(p)
            e.획_끝()
        최대.append(float(np.abs(py.V - cc.V).max()))
    for e in (py, cc):
        e.획_시작("잡기", 15, 1.0, True, False)
        for p in ([10, -50, 0], [12, -55, 3], [14, -60, 6]):
            e.점_더하기(p)
        e.획_끝()
    최대.append(float(np.abs(py.V - cc.V).max()))
    print("붓질마다 파이썬 대 C++ 정점 최대 차이 (mm):", ["%.1e" % x for x in 최대])
    assert max(최대) < 1e-6, 최대
    assert [g["자국수"] for g in py.저널] == [g["자국수"] for g in cc.저널], "자국 수가 같아야 한다"
    r = 재생(cc.저널, V, F)
    assert r.해시() == cc.해시(), "C++ 재생 = C++ 화면"
    움직임 = float(np.abs(cc.V - V).max())
    assert 움직임 > 1.0, "붓이 먹어야 한다"
    print("검사 통과 — 파이썬 = C++ (최대 %.1e mm) · 자국 수 같음 · 재생 해시 같음 · 최대 변위 %.1f mm" % (max(최대), 움직임))

    # 속도 — 큰 메시
    p = r"G:\art2real\원화3d\out\결과물\오메가몬\결과.ply"
    if os.path.exists(p):
        import trimesh
        m = trimesh.load(p, force="mesh", process=True)
        V = np.asarray(m.vertices); F = np.asarray(m.faces)
        V = (V - (V.max(0) + V.min(0)) / 2) * (150 / np.ptp(V[:, 2]))
        c = V[np.argmin(V[:, 1])]
        for 이름, cls in (("파이썬", E.조각), ("C++", 조각C)):
            e = cls(V, F)
            e.획_시작("그리기", 6, 0.5)
            t = time.perf_counter()
            for k in range(20):
                e.점_더하기(c + [k * 0.5, 0, 0])
            s = e.획_끝()
            print("  %-4s 정점 %d · 자국 %d · 자국당 %.2f ms" % (이름, len(V), s["자국수"], 1000 * (time.perf_counter() - t) / s["자국수"]))


if __name__ == "__main__":
    검사()
