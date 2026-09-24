"""대칭 기본형 소체 — 계획 단계 1 (09-25, `claudedocs/plan_기본형키트_20260925.md`).

    python 조각/소체.py 검사

  사용자 실물 출력(09-25): 「대칭성이 없음 · 무슨 부품인지 모름」 → 「기본 도형이라면 대칭성이 무조건 보장」.
  켜부피(복원)는 참고 원본으로만 두고, 출력할 몸은 **이름 붙은 기본형(로프트)** 으로 짠다:
    머리 · 가슴 · 골반 — 몸 중심선(x = 0) 위 대칭 로프트(높이마다 타원 · 가운데가 중심선).
    윗팔 · 아래팔 · 허벅지 · 정강이(.왼 · .오) — 왼 · 오 의 **굵기 곡선 · 길이를 평균한 한 파라미터**를 나눠 쓴다.
  자세만 따로: 왼 · 오 방향이 거울상에 가까우면(12° 안) 완전한 거울, 아니면 각자 방향(검 들기 · 팔 앞뒤).
  관절(팔꿈치 · 무릎)은 부위 길이의 가운데 — 두 로프트가 같은 단면에서 만난다(평면 이음면, 단계 2).
  사람형이 아니면(팔 · 다리를 못 찾으면) None.
"""
import numpy as np

import 그림 as G

매듭 = 5
거울문턱 = 12.0                                                          # 도


def _중심(앞, fa, L):
    ys, xs = np.nonzero(L == 2)
    return (np.median(xs) + 0.5 - fa.c) * fa.s if len(xs) else 0.0


def _세로(앞, 옆, fa, fo, 칸, x0, 매듭수=매듭):
    """세로 부위(머리 · 몸통 조각) -> 중심선 위 대칭 로프트 (점, w, d). 칸 = bool 화소."""
    rows = np.nonzero(칸.any(1))[0]
    if len(rows) < 3:
        return None
    zs = (fa.bot - np.linspace(rows.max() + 1, rows.min(), 매듭수)) * fa.s
    점, w, d = [], [], []
    for z in zs:
        r = int(np.clip(round(fa.bot - z / fa.s - 0.5), rows.min(), rows.max()))
        xs = np.nonzero(칸[r])[0]
        if not len(xs):
            continue
        x_lo, x_hi = (xs.min() - fa.c) * fa.s, (xs.max() + 1 - fa.c) * fa.s
        반 = max(x_hi - x0, x0 - x_lo) * 0.5 + (x_hi - x_lo) * 0.25       # 좌우 반폭의 평균에 가깝게(한쪽 쏠림을 줄임)
        옆값 = G._옆줄(옆, fo, z) or (0.0, 2 * 반)
        점.append([0.0, round(float(옆값[0]), 1), round(float(z), 1)])
        w.append(round(float(2 * 반), 1))
        d.append(round(float(옆값[1]), 1))
    return 점, w, d


def _팔다리(앞, 옆, fa, fo, L, 번호, 몸중심, 세로, 매듭수=2 * 매듭 - 1):
    """팔 · 다리 한쪽 -> (뿌리 [x, z], 방향 단위벡터, 길이, 굵기 곡선 w[매듭수], 깊이 곡선 d[매듭수])."""
    ys, xs = np.nonzero(L == 번호)
    if len(ys) < 30:
        return None
    X, Z = (xs + 0.5 - fa.c) * fa.s, (fa.bot - ys - 0.5) * fa.s
    Q = np.stack([X, Z], 1)
    c = Q - Q.mean(0)
    a = np.linalg.eigh(c.T @ c)[1][:, -1]
    if (Q.mean(0) - 몸중심) @ a < 0:                                        # 뿌리(몸 쪽)에서 끝으로
        a = -a
    nrm = np.array([-a[1], a[0]])
    t, q = Q @ a, Q @ nrm
    lo, hi = t.min(), t.max()
    띠 = max((hi - lo) / (2 * (매듭수 - 1)), fa.s)
    w, d, 중 = [], [], []
    for tk in np.linspace(lo, hi, 매듭수):
        m = np.abs(t - tk) <= 띠
        if not m.any():
            m = np.abs(t - tk) <= np.abs(t - tk).min() + fa.s
        너비 = q[m].max() - q[m].min() + fa.s
        중.append((q[m].max() + q[m].min()) / 2)
        z = (tk * a + 중[-1] * nrm)[1]
        옆값 = G._옆줄(옆, fo, z)
        w.append(너비)
        d.append(옆값[1] if (세로 and 옆값) else 너비)
    qc = float(np.median(중))
    뿌리 = lo * a + qc * nrm
    return {"뿌리": 뿌리, "방향": a, "길이": hi - lo, "w": np.array(w), "d": np.array(d),
            "y": (G._옆줄(옆, fo, float(뿌리[1])) or (0.0, 0.0))[0]}


def _거울평균(왼, 오, x0):
    """왼 · 오 -> 같은 모양(굵기 · 길이 평균) · 자세는 거울에 가까우면 거울, 아니면 각자."""
    w = np.round((왼["w"] + 오["w"]) / 2, 1)
    d = np.round((왼["d"] + 오["d"]) / 2, 1)
    길이 = (왼["길이"] + 오["길이"]) / 2
    거울 = lambda v: np.array([-v[0], v[1]])
    각 = np.degrees(np.arccos(np.clip(왼["방향"] @ 거울(오["방향"]), -1, 1)))
    if 각 < 거울문턱:
        방향 = 왼["방향"] + 거울(오["방향"])
        방향 = 방향 / np.linalg.norm(방향)
        dx = ((왼["뿌리"][0] - x0) + (x0 - 오["뿌리"][0])) / 2
        z = (왼["뿌리"][1] + 오["뿌리"][1]) / 2
        y = (왼["y"] + 오["y"]) / 2
        return ({"뿌리": np.array([x0 + dx, z]), "방향": 방향, "y": y}, {"뿌리": np.array([x0 - dx, z]), "방향": 거울(방향), "y": y},
                w, d, 길이, True)
    return 왼, 오, w, d, 길이, False


def _로프트두(쪽, w, d, 길이, x0, 이름위, 이름아래, 옆쪽):
    """한쪽 팔다리 -> 윗 · 아래 두 로프트(가운데 매듭에서 만남)."""
    n = len(w)
    ts = np.linspace(0, 길이, n)
    점 = [[round(float(쪽["뿌리"][0] + t * 쪽["방향"][0] - x0), 1), round(float(쪽["y"]), 1),
          round(float(쪽["뿌리"][1] + t * 쪽["방향"][1]), 1)] for t in ts]
    h = n // 2
    return ['로프트(점=%s, w=%s, d=%s, 이름="%s.%s")' % (점[:h + 1], list(map(float, w[:h + 1])), list(map(float, d[:h + 1])), 이름위, 옆쪽),
            '로프트(점=%s, w=%s, d=%s, 이름="%s.%s")' % (점[h:], list(map(float, w[h:])), list(map(float, d[h:])), 이름아래, 옆쪽)]


def 소체(앞, 옆, 키):
    """-> (DSL 줄, 정보) 또는 None. 부품 이름: 머리 · 가슴 · 골반 · 윗팔 · 아래팔 · 허벅지 · 정강이 (.왼 · .오)."""
    fa, fo = G.틀(앞, 키), G.틀(옆, 키)
    L = G.부위나누기(앞)
    if not ((L == 3).sum() >= 30 and (L == 4).sum() >= 30 and (L == 5).sum() >= 30 and (L == 6).sum() >= 30):
        return None
    x0 = _중심(앞, fa, L)
    몸 = (L == 2) | (L == 7)
    rows = np.nonzero(몸.any(1))[0]
    폭 = np.array([np.count_nonzero(몸[r]) for r in rows])
    가운데 = (rows >= rows.min() + 0.3 * len(rows)) & (rows <= rows.min() + 0.7 * len(rows))
    허리 = rows[가운데][np.argmin(폭[가운데])] if 가운데.any() else rows[len(rows) // 2]
    가슴칸, 골반칸 = 몸.copy(), 몸.copy()
    가슴칸[허리 + 1:] = False
    골반칸[:허리] = False
    토막 = []
    for 이름, 칸 in (("머리", L == 1), ("가슴", 가슴칸), ("골반", 골반칸)):
        r = _세로(앞, 옆, fa, fo, 칸, x0)
        if r:
            점, w, d = r
            점 = [[0.0, p[1], p[2]] for p in 점]
            토막.append('로프트(점=%s, w=%s, d=%s, 이름="%s")' % (점, w, d, 이름))
    몸중심 = np.array([x0, (fa.bot - np.median(np.nonzero(몸)[0])) * fa.s])
    정보 = {"중심 x": round(float(x0), 2), "허리 z": round(float((fa.bot - 허리) * fa.s), 1)}
    for (왼번, 오번, 위, 아래, 세로) in ((3, 4, "윗팔", "아래팔", False), (5, 6, "허벅지", "정강이", True)):
        왼 = _팔다리(앞, 옆, fa, fo, L, 왼번, 몸중심, 세로)
        오 = _팔다리(앞, 옆, fa, fo, L, 오번, 몸중심, 세로)
        if not (왼 and 오):
            return None
        a, b, w, d, 길이, 거울 = _거울평균(왼, 오, x0)
        정보[위] = {"거울 자세": 거울, "길이": round(float(길이), 1)}
        토막 += _로프트두(a, w, d, 길이, x0, 위, 아래, "왼")
        토막 += _로프트두(b, w, d, 길이, x0, 위, 아래, "오")
    return " + ".join(토막), 정보


def 부품들(줄):
    """DSL 줄 -> {이름: (V, F)} (로프트 이름으로 나눔). 단계 2 에서 이음면 · 다보를 붙일 단위."""
    import ast
    import dsl as D
    out = {}
    for t in D._항들(ast.parse(줄, mode="eval").body):
        이름 = next(k.value.value for k in t.keywords if k.arg == "이름")
        out[이름] = D.메시(D._풀기(t))
    return out


def 거울오차(부):
    """짝 부품(.왼 · .오) 부피 차 / 부피 — 모양이 같은가(자세와 무관)."""
    import trimesh
    오차 = {}
    for k in 부:
        if k.endswith(".왼") and k[:-2] + ".오" in 부:
            a = trimesh.Trimesh(*부[k], process=False).volume
            b = trimesh.Trimesh(*부[k[:-2] + ".오"], process=False).volume
            오차[k[:-2]] = round(abs(a - b) / max(a, b), 5)
    return 오차


def 검사():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "벤치"))
    import 어댑터 as A
    시트 = os.path.join(A.원화3d, "out", "시트")
    for c in ("마네킹_차렷", "마네킹_T포즈", "마네킹_팔앞뒤", "avatarsample_d"):
        앞, 옆 = (G.마스크(os.path.join(시트, c, n + ".png")) for n in ("front", "side"))
        r = 소체(앞, 옆, 150.0)
        assert r, c
        부 = 부품들(r[0])
        e = 거울오차(부)
        assert all(v < 1e-3 for v in e.values()), (c, e)
        print("  %-16s 부품 %d · 거울 오차 %s · %s" % (c, len(부), max(e.values()), {k: v for k, v in r[1].items() if k in ("윗팔", "허벅지")}))
    print("소체 검사 통과 — 짝 부품 모양 같음")


if __name__ == "__main__":
    검사()
