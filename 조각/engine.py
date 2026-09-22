# -*- coding: utf-8 -*-
"""**조각 엔진** — 붓 다섯 개 · 세계 좌표 기록 · 재생하면 같은 결과 (2026-09-22).

    python engine.py 검사                     자체검사 (결정성 · 쪼개 넣기 무관 · 대칭 · 매끈)
    python engine.py 재생 <저널.json>          저널을 새 엔진으로 처음부터 다시 틀고 해시를 찍는다 (= 「스크립트」 쪽)

사용자(09-22): 「스컬프트에 대한 프로세스를 exe 로 구현해 볼래? 내가 테스트해 볼 수 있게」
조사: `claudedocs/research_스컬프트엔진_20260922.md` — 붓 공식은 SculptGL 그대로.

**GUI 와 스크립트가 같은 결과를 내는 이유는 구조다.** 화면(ui/)은 마우스를 모델 위 **세계 좌표 점**으로 바꿔
`획_시작 · 점_더하기 · 획_끝` 을 부를 뿐이고, 붓 계산은 전부 여기서 한다. 저널에는 그 점들이 그대로 남는다.
재생도 같은 세 함수를 **점 하나씩** 부른다 — 점을 몇 개씩 묶어 넣어도 결과가 같게 짰다(간격 나머지를 이어 간다).

    폴오프    f = 3d⁴ − 4d³ + 1       (d = 거리/반지름 · 중심 1 · 가장자리 0)
    간격      자국 사이 = 0.15 × 반지름 (세계 길이 — 화면 픽셀이 아니다)
    그리기    v += n̄ · (I·R·0.1) · f            n̄ = 반지름 안 정점 법선의 평균 (Blender Draw)
    부풀리기  v += n_v · (I·R·0.1) · f           정점마다 자기 법선
    매끈      v += (이웃 평균 − v) · I · f
    잡기      획 시작 때 반지름 안 정점과 원래 위치 v₀ 를 잡고, v = v₀ + (지금 점 − 시작 점) · f  (누적 없음)
    주름      v += (c − v) · f · 0.5·I  −  n̄ · f⁵ · (I·R·0.07)     기본이 판다 (반전하면 솟는다)
    반전      그리기 · 부풀리기 · 주름은 부호를 뒤집는다 (Ctrl)
    뒷면 막기  법선이 n̄ 과 반대인 정점(얇은 판 뒤쪽)은 안 건드린다 — 화면이 아니라 표면으로 판단한다
    대칭 X    자국마다 x 를 뒤집은 자국을 하나 더 찍는다
"""
import hashlib
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

붓들 = ("그리기", "부풀리기", "매끈", "잡기", "주름")
간격비 = 0.15


def 폴오프(d):
    d = np.clip(d, 0.0, 1.0)
    return 3 * d ** 4 - 4 * d ** 3 + 1


def 기본메시(세분=6, 반지름=50.0):
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=세분, radius=반지름)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


class 조각:
    def __init__(self, V, F):
        self.V0 = np.array(V, np.float64)
        self.F = np.array(F, np.int64)
        self.V = self.V0.copy()
        n = len(self.V)
        # 이웃 (매끈 붓) — CSR
        e = np.concatenate([self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]])
        e = np.concatenate([e, e[:, ::-1]])
        e = np.unique(e, axis=0)
        self.이웃시작 = np.searchsorted(e[:, 0], np.arange(n + 1))
        self.이웃 = e[:, 1]
        self.이웃수 = np.diff(self.이웃시작).astype(np.float64)
        self.저널 = []
        self._획 = None

    # ---------------------------------------------------------- 기하
    def 법선(self):
        V, F = self.V, self.F
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        vn = np.zeros_like(V)
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)

    def 해시(self):
        """정점 위치만 (위상은 안 바뀐다) · 1 µm 로 반올림 — 면 순서와 무관."""
        return hashlib.sha1(np.round(self.V, 3).tobytes()).hexdigest()[:12]

    # ---------------------------------------------------------- 획
    def 획_시작(self, 붓, 반지름, 세기, 대칭=False, 반전=False):
        assert 붓 in 붓들, 붓
        self._획 = {"붓": 붓, "반지름": float(반지름), "세기": float(세기), "대칭": bool(대칭), "반전": bool(반전),
                   "점": [], "자국수": 0, "_마지막": None, "_남은": 0.0, "_잡기": None}
        return self._획

    def 점_더하기(self, p):
        """세계 좌표 점 하나. 마지막 자국에서 간격만큼씩 걸어가며 자국을 찍는다. 바뀐 정점 번호를 돌려준다."""
        s = self._획
        p = np.asarray(p, np.float64)
        s["점"].append([round(float(x), 4) for x in p])
        p = np.asarray(s["점"][-1])                               # 저널에 남는 값과 똑같은 값으로 계산한다
        바뀐 = set()
        if s["붓"] == "잡기":
            if s["_잡기"] is None:
                s["_잡기"] = self._잡기_준비(p)
            바뀐 |= self._잡기(p)
            s["자국수"] += 1
            return sorted(바뀐)
        if s["_마지막"] is None:
            바뀐 |= self._자국(p)
            s["_마지막"] = p
            return sorted(바뀐)
        간 = 간격비 * s["반지름"]
        a = s["_마지막"]                                           # 직전 점 (자국이 아니라 입력 점)
        L = float(np.linalg.norm(p - a))
        t = 간 - s["_남은"]                                        # _남은 = 마지막 자국에서 a 까지 걸은 거리
        while L > 0 and t <= L:
            바뀐 |= self._자국(a + (p - a) * (t / L))
            t += 간
        s["_남은"] = L - (t - 간)                                  # 자국이 없었으면 _남은 + L 과 같다
        s["_마지막"] = p
        return sorted(바뀐)

    def 획_끝(self):
        s = self._획
        self._획 = None
        if s is None or not s["점"]:
            return None
        기록 = {k: v for k, v in s.items() if not k.startswith("_")}
        기록["해시"] = self.해시()
        self.저널.append(기록)
        return 기록

    # ---------------------------------------------------------- 자국
    def _모으기(self, c, R):
        tree = cKDTree(self.V)
        idx = np.array(sorted(tree.query_ball_point(c, R)), np.int64)
        return idx

    def _자국(self, c):
        s = self._획
        중심들 = [c] + ([c * [-1, 1, 1]] if s["대칭"] else [])
        N = self.법선()
        움직임 = [self._한자국(np.asarray(q, np.float64), N) for q in 중심들]   # 같은 상태에서 다 계산하고
        바뀐 = set()
        for idx, dv in 움직임:                                                  # 한꺼번에 더한다 (대칭이 정확해진다)
            np.add.at(self.V, idx, dv)
            바뀐 |= set(idx[np.any(dv != 0, axis=1)].tolist())
        s["자국수"] += 1
        return 바뀐

    def _한자국(self, c, N_all):
        """이 자국이 옮길 (정점 번호, 변위) — 아직 더하지 않는다."""
        s = self._획
        R, I = s["반지름"], s["세기"]
        부호 = -1.0 if s["반전"] else 1.0
        idx = self._모으기(c, R)
        if len(idx) == 0:
            return idx, np.zeros((0, 3))
        P = self.V[idx]
        d = np.linalg.norm(P - c, axis=1) / R
        f = 폴오프(d)
        N = N_all[idx]
        n̄ = (N * f[:, None]).sum(0)
        n̄ = n̄ / max(np.linalg.norm(n̄), 1e-12)
        앞 = (N @ n̄) > 0                                           # 뒷면 막기
        f = f * 앞
        붓 = s["붓"]
        if 붓 == "그리기":
            dv = n̄ * (I * R * 0.1 * 부호) * f[:, None]
        elif 붓 == "부풀리기":
            dv = N * (I * R * 0.1 * 부호) * f[:, None]
        elif 붓 == "매끈":
            평균 = np.add.reduceat(self.V[self.이웃], self.이웃시작[:-1])[idx] / self.이웃수[idx, None]
            dv = (평균 - P) * (I * f)[:, None]
        else:  # 주름 — 기본이 **판다**(Blender 와 같다) · 반전하면 솟는다
            dv = (c - P) * (f * 0.5 * I)[:, None] - n̄ * (I * R * 0.07 * 부호) * (f ** 5)[:, None]
        return idx, dv

    def _잡기_준비(self, p):
        s = self._획
        잡 = []
        for q, 거울 in [(p, 1.0)] + ([(p * [-1, 1, 1], -1.0)] if s["대칭"] else []):
            idx = self._모으기(q, s["반지름"])
            f = 폴오프(np.linalg.norm(self.V[idx] - q, axis=1) / s["반지름"])
            잡.append((idx, self.V[idx].copy(), f, 거울))
        return {"시작": p.copy(), "잡": 잡}

    def _잡기(self, p):
        g = self._획["_잡기"]
        δ = (p - g["시작"]) * self._획["세기"]
        바뀐 = set()
        for idx, v0, f, 거울 in g["잡"]:
            dd = δ * np.array([거울, 1.0, 1.0])
            self.V[idx] = v0 + dd * f[:, None]
            바뀐 |= set(idx.tolist())
        return 바뀐

    # ---------------------------------------------------------- 저널
    def 되감기(self, k):
        """처음부터 k 획까지 다시 튼다 (ponytail: 매번 처음부터 — 느려지면 스냅숏)."""
        옛 = self.저널[:k]
        self.V = self.V0.copy()
        self.저널 = []
        for 획 in 옛:
            재생한획(self, 획)
        return self.해시()


def 재생한획(엔진, 획):
    엔진.획_시작(획["붓"], 획["반지름"], 획["세기"], 획["대칭"], 획["반전"])
    for p in 획["점"]:
        엔진.점_더하기(p)
    return 엔진.획_끝()


def 재생(저널, V, F):
    e = 조각(V, F)
    for 획 in 저널:
        재생한획(e, 획)
    return e


def 검사():
    V, F = 기본메시(세분=4)                                    # 2562 정점 — 빠르게
    선 = [[x, -49.0, 5.0] for x in np.linspace(-20, 20, 9)]

    def 한번(묶음):
        e = 조각(V, F)
        e.획_시작("그리기", 12, 0.5)
        for i in range(0, len(선), 묶음):
            for p in 선[i:i + 묶음]:
                e.점_더하기(p)
        e.획_끝()
        e.획_시작("부풀리기", 10, 0.3, 대칭=True); [e.점_더하기(p) for p in 선[:4]]; e.획_끝()
        e.획_시작("잡기", 15, 1.0); [e.점_더하기(p) for p in ([0, -50, 0], [0, -58, 3])]; e.획_끝()
        e.획_시작("주름", 8, 0.5, 반전=True); [e.점_더하기(p) for p in 선]; e.획_끝()
        return e

    a, b = 한번(1), 한번(1)
    assert a.해시() == b.해시(), "결정성"
    assert a.해시() != 조각(V, F).해시(), "붓이 먹어야 한다"
    c = 재생(json.loads(json.dumps(a.저널)), V, F)                # 저장했다 읽어 다시 틀기
    assert c.해시() == a.해시(), ("재생", c.해시(), a.해시())
    # 대칭: 대칭 붓 한 번이면 x 거울상과 같아야 한다 (기본 구가 x 거울 대칭)
    e = 조각(V, F)
    e.획_시작("그리기", 12, 0.5, 대칭=True); [e.점_더하기(p) for p in 선[:3]]; e.획_끝()
    거울 = e.V * [-1, 1, 1]
    dd, _ = cKDTree(e.V).query(거울)
    assert dd.max() < 1e-6, ("대칭", dd.max())
    # 매끈: 울퉁불퉁을 줄인다
    e = 조각(V, F)
    rng = np.random.default_rng(0)
    e.V += rng.normal(0, 0.5, e.V.shape)
    def 거칠기(e):
        m = np.array([e.V[e.이웃[e.이웃시작[i]:e.이웃시작[i + 1]]].mean(0) for i in range(len(e.V))])
        return float(np.linalg.norm(e.V - m, axis=1).mean())
    r0 = 거칠기(e)
    e.획_시작("매끈", 60, 0.8); [e.점_더하기(p) for p in 선]; e.획_끝()
    assert 거칠기(e) < r0, "매끈"
    # 되감기: 2 획으로 되감으면 2 획까지만 튼 것과 같다
    a.되감기(2)
    d = 재생(a.저널, V, F)
    assert d.해시() == a.해시()
    print("검사 통과 — 결정성 · 재생 · 대칭 · 매끈 · 되감기")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "검사":
        검사()
    elif a and a[0] == "재생":
        j = json.load(open(a[1], encoding="utf-8"))
        V, F = (np.array(j["기본"]["V"]), np.array(j["기본"]["F"])) if "기본" in j else 기본메시()
        e = 재생(j["저널"], V, F)
        print("재생 해시", e.해시(), "· 저널 끝 해시", j["저널"][-1]["해시"] if j["저널"] else "-",
              "· 같음" if j["저널"] and e.해시() == j["저널"][-1]["해시"] else "")
