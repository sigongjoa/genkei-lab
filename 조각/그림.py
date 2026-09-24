"""그림 → 형태. 앞 · 옆 그림 두 장과 키(mm)만 보고 카탈로그에서 형태를 고른다. 정답은 모른다.

    python 조각/그림.py 검사

입력 규약: 흰(또는 투명) 바탕에 물체 하나. 앞 그림은 정면(오른쪽 = +x), 옆 그림은 **물체가 오른쪽을 본다**.
두 그림 다 물체 키를 `키` mm 로 잡는다 — 그래서 그림마다 해상도가 달라도 된다.

  1) 잰다      물체 범위 · 줄마다 폭 · 연결 성분 · 안쪽 구멍
  2) 후보      카탈로그 항목마다 잰 수치로 DSL 한 줄 (상자 · 원기둥 · 원뿔 · 구 · 45도 상자 · 토러스 · 두 덩어리 ·
               사람형: 부위 타원뿔대(A) · 부위 로프트 5 단면(B) · 부위 없는 층 타원(C))
  3) 대 본다   후보 메시를 **입력 그림과 같은 틀**에 그려 앞 · 옆 실루엣 IoU 평균 — 이게 앱이 아는 유일한 점수
  4) 고른다    가장 높은 것. 0.005 안의 후보는 **동점** — 그림 두 장으로는 못 가른다. 동점 안 1순위는 카탈로그 순서, 나머지는 사람이 고를 몫
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import dsl as D

동점폭 = 0.005


def 마스크(경로):
    im = np.asarray(Image.open(경로).convert("RGBA"))
    return im[..., 3] > 127 if im[..., 3].min() < 255 else im[..., :3].mean(-1) < 250


class 틀:
    """그림 한 장의 화소 ↔ mm. 물체 키 = 키 mm, 가로 가운데 = 0, 바닥 = 0."""

    def __init__(self, m, 키):
        ys, xs = np.nonzero(m)
        self.m, self.s = m, 키 / (ys.max() + 1 - ys.min())            # mm / 화소
        self.c, self.bot = (xs.min() + xs.max() + 1) / 2, ys.max() + 1

    def 범위(self, m=None):
        """-> (가로 lo, hi, 높이 lo, hi) mm. 화소 가장자리로."""
        ys, xs = np.nonzero(self.m if m is None else m)
        u = lambda c: (c - self.c) * self.s
        z = lambda r: (self.bot - r) * self.s
        return u(xs.min()), u(xs.max() + 1), z(ys.max() + 1), z(ys.min())

    def 그리기(self, U, Zs, F):
        """메시를 이 그림 틀에 그린다 (정사영). U = 가로 mm, Zs = 높이 mm.
        PIL 다각형은 테두리를 포함해 칠해 1 화소쯤 뚱뚱하다 — 4 배로 그려 반 넘게 찬 화소만 센다."""
        k = 4
        px = (self.c + np.asarray(U) / self.s) * k
        py = (self.bot - np.asarray(Zs) / self.s) * k
        H, W = self.m.shape
        im = Image.new("1", (W * k, H * k), 0)
        d = ImageDraw.Draw(im)
        P = np.stack([px, py], 1) - 0.5                            # 부분 화소 중심에 맞춘다
        for t in np.asarray(F):
            d.polygon([tuple(P[i]) for i in t], fill=1)
        return np.asarray(im, np.float32).reshape(H, k, W, k).mean((1, 3)) > 0.5


def _iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def 후보들(앞, 옆, 키):
    """-> [(카탈로그 이름, DSL 줄)]. 앞 · 옆 = bool 마스크."""
    fa, fo = 틀(앞, 키), 틀(옆, 키)
    x0, x1, z0, z1 = fa.범위()
    y0, y1 = fo.범위()[:2]
    w, d, h = x1 - x0, y1 - y0, z1 - z0
    cx, cy = (x0 + x1) / 2, -(y0 + y1) / 2                       # 옆 그림 오른쪽 = 정면 = -y
    r1 = lambda v: round(float(v), 1)
    놓기 = "x=%s, y=%s, z0=%s" % (r1(cx), r1(cy), r1(z0))
    out = [("상자", "상자(w=%s, d=%s, h=%s, %s)" % (r1(w), r1(d), r1(h), 놓기)),
           ("원기둥", "원기둥(r=%s, h=%s, %s)" % (r1((w + d) / 4), r1(h), 놓기))]
    rows = np.nonzero(앞.any(1))[0]                               # 원뿔대: 높이마다 반폭에 직선
    k, b = np.polyfit((fa.bot - rows - 0.5) * fa.s, 앞[rows].sum(1) * fa.s / 2, 1)
    out.append(("원뿔", "원뿔(r아래=%s, r위=%s, h=%s, %s)" % (r1(max(k * z0 + b, 0.1)), r1(max(k * z1 + b, 0.1)), r1(h), 놓기)))
    out.append(("구", "구(r=%s, %s)" % (r1(h / 2), 놓기)))
    s45 = (w + d) / 2 / np.sqrt(2)
    out.append(("45도 상자", "상자(w=%s, d=%s, h=%s, %s, rz=45)" % (r1(s45), r1(s45), r1(h), 놓기)))
    out.append(("타원체", "타원체(w=%s, d=%s, h=%s, %s)" % (r1(w), r1(d), r1(h), 놓기)))
    out.append(("캡슐", "캡슐(r=%s, h=%s, %s)" % (r1((w + d) / 4), r1(h), 놓기)))
    out.append(("반구", "반구(w=%s, d=%s, h=%s, %s)" % (r1(w), r1(d), r1(h), 놓기)))
    (wa, wb), (da, db) = _곧은폭(앞, fa, z0, z1), _곧은폭(옆, fo, z0, z1)
    out.append(("각뿔대", "각뿔대(w아래=%s, w위=%s, d아래=%s, d위=%s, h=%s, %s)" % (r1(wa), r1(wb), r1(da), r1(db), r1(h), 놓기)))
    모 = (_모서리(앞, fa) + _모서리(옆, fo)) / 2
    if 모 > 1.0:
        out.append(("둥근상자", "둥근상자(w=%s, d=%s, h=%s, r=%s, %s)" % (r1(w), r1(d), r1(h), r1(모), 놓기)))
    for 축, m, f, 길이 in (("y", 앞, fa, d), ("x", 옆, fo, w)):       # 구멍이 보이는 판 = 토러스 · 관 축 방향
        hole = ndimage.binary_fill_holes(m) & ~m
        if hole.sum() > 50:
            hz = f.범위(hole)
            Ro, Ri = h / 2, (hz[3] - hz[2]) / 2
            out.append(("토러스", '토러스(R=%s, r=%s, 축="%s", %s)' % (r1((Ro + Ri) / 2), r1((Ro - Ri) / 2), 축, 놓기)))
            out.append(("관", '관(R=%s, r안=%s, 길이=%s, 축="%s", %s)' % (r1(Ro), r1(Ri), r1(길이), 축, 놓기)))
    la, na = ndimage.label(앞)
    lo, no = ndimage.label(옆)
    if na == 2 and no in (1, 2):                                 # 두 덩어리 — 짝짓기가 여럿일 수 있다
        xs = sorted((fa.범위(la == i) for i in (1, 2)), key=lambda b: b[0])
        ys = sorted((fo.범위(lo == i) for i in range(1, no + 1)), key=lambda b: b[0])
        기둥 = lambda bx, by: "상자(w=%s, d=%s, h=%s, x=%s, y=%s, z0=%s)" % (
            r1(bx[1] - bx[0]), r1(by[1] - by[0]), r1(bx[3] - bx[2]), r1((bx[0] + bx[1]) / 2), r1(-(by[0] + by[1]) / 2), r1(bx[2]))
        짝들 = [[(0, 0), (1, 0)]] if no == 1 else [[(0, 0), (1, 1)], [(0, 1), (1, 0)]]
        for 짝 in 짝들:
            out.append(("두 덩어리", " + ".join(기둥(xs[i], ys[j]) for i, j in 짝)))
    for 이름, 줄 in (("부위 타원뿔대", 사람형(앞, 옆, 키, 2)), ("부위 로프트", 사람형(앞, 옆, 키, 5)), ("혼합", 혼합(앞, 옆, 키)), ("혼합 겹침", 혼합겹침(앞, 옆, 키)),
                     ("층 타원", 층타원(앞, 옆, 키)), ("층 겹침", 층겹침(앞, 옆, 키))):
        if 줄:
            out.append((이름, 줄))
    return out


# ── 사람형 — 부위 나누기 (규칙) ────────────────────────────────────────────────
# 부위 번호: 1 머리 · 2 몸통 · 3 팔.왼 · 4 팔.오 · 5 다리.왼 · 6 다리.오 · 7 하체(다리가 안 갈라질 때)
# 왼 · 오 = 캐릭터 기준 (정면 -y 를 보니 +x 가 캐릭터 왼쪽)
부위이름 = {1: "머리", 2: "몸통", 3: "팔.왼", 4: "팔.오", 5: "다리.왼", 6: "다리.오", 7: "하체"}
세로부위 = {1, 2, 5, 6, 7}


def _줄조각(row):
    """한 줄의 연속 구간들 -> [(c0, c1)] (c1 제외)."""
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def 부위나누기(앞):
    """앞 마스크 -> 부위 번호 그림(같은 크기, 0 = 빈 곳). 사람형이 아니면 몸통 · 하체만 나온다.
    다리 = 가운데 세로줄이 비는 줄들(가랑이 아래) · 머리 = 위 8~35% 에서 가운데 구간이 가장 좁은 줄(몸통 폭의 0.8 미만일 때) 위 ·
    팔 = 가운데 구간에서 몸통 폭 밖으로 나온 곳과 가운데에 안 닿은 구간."""
    H, W = 앞.shape
    L = np.zeros(앞.shape, np.uint8)
    rows = np.nonzero(앞.any(1))[0]
    top, bot = rows.min(), rows.max()
    키px = bot - top + 1
    cx = int(np.median(np.nonzero(앞)[1]))
    조각 = {r: _줄조각(앞[r]) for r in rows}
    가운데 = lambda r: next(((a, b) for a, b in 조각[r] if a <= cx < b), None)
    # 가랑이: 아래 60% 에서 가운데 세로줄이 빈 줄이 가장 길게 이어진 덩어리의 맨 윗줄. 발끼리 붙어 맨 아래 줄이
    # 차 있어도(차렷) 찾는다. 그 덩어리가 키의 5% 미만이면 다리가 안 갈라진 것.
    가랑이, 긴 = bot + 1, 0
    r = bot
    while r >= top + 0.4 * 키px:
        if 가운데(r) is None:
            r0 = r
            while r >= top and 가운데(r) is None:
                r -= 1
            if r0 - r > 긴:
                긴, 가랑이 = r0 - r, r + 1
        r -= 1
    if 긴 < 0.05 * 키px:
        가랑이 = bot + 1
    폭 = lambda r: (lambda g: g[1] - g[0] if g else 0)(가운데(r))
    몸통줄 = [r for r in range(int(top + 0.35 * 키px), min(가랑이, bot + 1)) if 가운데(r)]
    몸통폭 = np.median([폭(r) for r in 몸통줄]) if 몸통줄 else W
    띠 = [r for r in range(int(top + 0.08 * 키px), int(top + 0.35 * 키px)) if 가운데(r)]
    목 = min(띠, key=폭) if 띠 else top
    if not 띠 or 폭(목) >= 0.8 * 몸통폭:
        목 = top                                                       # 머리를 못 찾았다
    반 = 몸통폭 / 2 * 1.1
    for r in rows:
        for a, b in 조각[r]:
            if r < 목:
                L[r, a:b] = 1
            elif r >= 가랑이:
                continue
            elif a <= cx < b:
                lo, hi = int(max(a, cx - 반)), int(min(b, cx + 반))
                L[r, lo:hi] = 2
                L[r, a:lo] = 4
                L[r, hi:b] = 3
            else:
                L[r, a:b] = 3 if a > cx else 4
    if 가랑이 <= bot:
        for r in range(가랑이, bot + 1):
            조 = sorted(조각[r], key=lambda g: abs((g[0] + g[1]) / 2 - cx))
            for i, (a, b) in enumerate(조):
                if i < 2:
                    L[r, a:b] = 5 if (a + b) / 2 > cx else 6
                else:
                    L[r, a:b] = 3 if a > cx else 4
    else:
        L[(L == 2) & (np.arange(H)[:, None] > top + 0.55 * 키px)] = 7        # 다리가 안 갈라지면 아래는 하체
    for 팔 in (3, 4):                                                   # 몸통 옆구리가 중앙값보다 넓은 줄의 부스러기는 팔이 아니다
        lab, n = ndimage.label(L == 팔)
        if n > 1:
            큰 = 1 + np.argmax(np.bincount(lab.ravel())[1:])
            L[(lab > 0) & (lab != 큰)] = 2
    return L


def _옆줄(옆, fo, z):
    """높이 z(mm) 에서 옆 그림 한 줄의 (y 가운데, 깊이) mm. 그 줄이 비면 None."""
    r = int(np.clip(round(fo.bot - z / fo.s - 0.5), 0, 옆.shape[0] - 1))
    xs = np.nonzero(옆[r])[0]
    if not len(xs):
        return None
    y0, y1 = (xs.min() - fo.c) * fo.s, (xs.max() + 1 - fo.c) * fo.s
    return -(y0 + y1) / 2, y1 - y0


def _로프트줄(ys, xs, 번호, 앞, 옆, fa, fo, 매듭):
    """한 부위 화소 -> 로프트 DSL 한 토막. 세로 부위는 z 를, 팔은 앞 그림 주성분을 중심선으로."""
    X, Z = (xs + 0.5 - fa.c) * fa.s, (fa.bot - ys - 0.5) * fa.s
    Q = np.stack([X, Z], 1)
    if 번호 in 세로부위:
        a = np.array([0.0, 1.0])
    else:
        c = Q - Q.mean(0)
        a = np.linalg.eigh(c.T @ c)[1][:, -1]
    nrm = np.array([-a[1], a[0]])
    t, q = Q @ a, Q @ nrm
    lo, hi = t.min() - fa.s / 2, t.max() + fa.s / 2
    띠 = max((hi - lo) / (2 * (매듭 - 1)), fa.s)
    점, w, d = [], [], []
    for tk in np.linspace(lo, hi, 매듭):
        m = np.abs(t - tk) <= 띠
        if not m.any():
            m = np.abs(t - tk) <= np.abs(t - tk).min() + fa.s
        qc, 너비 = (q[m].max() + q[m].min()) / 2, q[m].max() - q[m].min() + fa.s
        x, z = tk * a + qc * nrm
        옆값 = _옆줄(옆, fo, z)
        if 번호 in 세로부위 and 옆값:
            y, 깊이 = 옆값
        else:                                                           # 팔은 옆에서 몸통과 겹친다 — 둥글다고 본다
            y, 깊이 = (옆값[0] if 옆값 else 0.0), 너비
        점.append([round(float(x), 1), round(float(y), 1), round(float(z), 1)])
        w.append(round(float(너비), 1))
        d.append(round(float(깊이), 1))
    return '로프트(점=%s, w=%s, d=%s, 이름="%s")' % (점, w, d, 부위이름[번호])


def 사람형(앞, 옆, 키, 매듭):
    """부위마다 로프트 — 매듭 2 = 부위별 타원뿔대(A), 5 = 부위별 5 단면(B)."""
    fa, fo = 틀(앞, 키), 틀(옆, 키)
    L = 부위나누기(앞)
    토막 = []
    for 번호 in sorted(부위이름):
        ys, xs = np.nonzero(L == 번호)
        if len(ys) < 30:
            continue
        토막.append(_로프트줄(ys, xs, 번호, 앞, 옆, fa, fo, 매듭))
    return " + ".join(토막) if 토막 else None


def 층타원(앞, 옆, 키, 층수=60):
    """부위 없이 높이마다 앞 구간 하나에 타원 단면 한 켜(C). 깊이 · y 는 같은 높이의 옆 한 줄."""
    fa, fo = 틀(앞, 키), 틀(옆, 키)
    rows = np.nonzero(앞.any(1))[0]
    top, bot = rows.min(), rows.max() + 1
    경계 = np.linspace(top, bot, 층수 + 1)
    토막 = []
    for r0, r1 in zip(경계[:-1], 경계[1:]):
        r = int((r0 + r1) / 2)
        z0, z1 = (fa.bot - r1) * fa.s, (fa.bot - r0) * fa.s + 0.01          # 켜끼리 살짝 겹친다
        옆값 = _옆줄(옆, fo, (z0 + z1) / 2)
        if not 옆값:
            continue
        y, 깊이 = 옆값
        for a, b in _줄조각(앞[r]):
            x, 너비 = ((a + b) / 2 - fa.c) * fa.s, (b - a) * fa.s
            토막.append("로프트(점=[[%.1f, %.1f, %.2f], [%.1f, %.1f, %.2f]], w=[%.1f, %.1f], d=[%.1f, %.1f])"
                      % (x, y, z0, x, y, z1, 너비, 너비, 깊이, 깊이))
    return " + ".join(토막) if 토막 else None


def _곧은폭(m, f, z0, z1):
    """높이마다 폭(화소 수)에 직선 -> (바닥 폭, 꼭대기 폭) mm. 각뿔대 재기."""
    rows = np.nonzero(m.any(1))[0]
    k, b = np.polyfit((f.bot - rows - 0.5) * f.s, m[rows].sum(1) * f.s, 1)
    return max(k * z0 + b, 0.0), max(k * z1 + b, 0.0)


def _모서리(m, f):
    """맨 위에서 몇 줄 내려가야 폭이 다 차나 -> 둥근 모서리 반지름 mm (네모면 0 근처)."""
    rows = np.nonzero(m.any(1))[0]
    폭 = m[rows].sum(1)
    k = int(np.argmax(폭 >= 폭.max() - 1))
    return k * f.s


def 혼합(앞, 옆, 키, 층수=60, 팔매듭=9):
    """층 타원(C) + 팔만 로프트. 켜마다 앞 구간을 부위 번호로 다시 쪼개 팔 조각은 빼고, 팔은 주성분 중심선을 따라
    단면 아홉 개(깊이 = 폭, 둥글다). T포즈 팔이 몸통 깊이의 납작한 판이 되던 것(옆 그림 겹침)을 피한다."""
    fa, fo = 틀(앞, 키), 틀(옆, 키)
    L = 부위나누기(앞)
    팔 = [번 for 번 in (3, 4) if (L == 번).sum() >= 30]
    if not 팔:
        return None
    rows = np.nonzero(앞.any(1))[0]
    경계 = np.linspace(rows.min(), rows.max() + 1, 층수 + 1)
    토막 = []
    for r0, r1 in zip(경계[:-1], 경계[1:]):
        r = int((r0 + r1) / 2)
        z0, z1 = (fa.bot - r1) * fa.s, (fa.bot - r0) * fa.s + 0.01
        옆값 = _옆줄(옆, fo, (z0 + z1) / 2)
        if not 옆값:
            continue
        y, 깊이 = 옆값
        for a, b in _줄조각(앞[r] & ~np.isin(L[r], 팔)):
            x, 너비 = ((a + b) / 2 - fa.c) * fa.s, (b - a) * fa.s
            토막.append("로프트(점=[[%.1f, %.1f, %.2f], [%.1f, %.1f, %.2f]], w=[%.1f, %.1f], d=[%.1f, %.1f])"
                      % (x, y, z0, x, y, z1, 너비, 너비, 깊이, 깊이))
    for 번 in 팔:
        ys, xs = np.nonzero(L == 번)
        토막.append(_로프트줄(ys, xs, 번, 앞, 옆, fa, fo, 팔매듭))
    return " + ".join(토막)


def 켜줄기(켜들):
    """켜들 = [(z0, z1, y, 깊이, [(x0, x1), ...]), ...] 아래→위. 위아래 구간이 1대1로 겹치면 한 줄기로 잇는다.
    -> [ [(켜 번호, 구간), ...], ... ]. 갈라지거나 합쳐지는 곳에서 줄기가 끊긴다(가랑이 · 어깨)."""
    줄기들, 열린 = [], {}
    for i, (_, _, _, _, 구간들) in enumerate(켜들):
        새열린 = {}
        아래 = list(열린.items())                                          # (구간 번호, 줄기)
        for j, g in enumerate(구간들):
            닿 = [(jj, 줄) for jj, 줄 in 아래 if min(g[1], 줄[-1][1][1]) > max(g[0], 줄[-1][1][0])]
            if len(닿) == 1:
                jj, 줄 = 닿[0]
                위짝 = [gg for gg in 구간들 if min(gg[1], 줄[-1][1][1]) > max(gg[0], 줄[-1][1][0])]
                if len(위짝) == 1:
                    줄.append((i, g))
                    새열린[j] = 줄
                    continue
            줄 = [(i, g)]
            줄기들.append(줄)
            새열린[j] = 줄
        열린 = 새열린
    return 줄기들


def 켜쌓기줄(켜들, fa):
    """줄기마다 층쌓기 한 토막. 단면은 켜 가운데 높이에, 줄기 두 끝은 켜 경계까지 같은 단면으로 늘인다."""
    토막 = []
    for 줄 in 켜줄기(켜들):
        z, x, y, w, d = [], [], [], [], []
        for n, (i, (a, b)) in enumerate(줄):
            z0, z1, yy, 깊이, _ = 켜들[i]
            xx, ww = ((a + b) / 2 - fa.c) * fa.s, (b - a) * fa.s
            높이들 = [z0 + 0.0] if n == 0 else []
            높이들 += [(z0 + z1) / 2]
            if n == len(줄) - 1:
                높이들 += [z1 + 0.01]
            for h in 높이들:
                z.append(round(float(h), 2)); x.append(round(float(xx), 1)); y.append(round(float(yy), 1)); w.append(round(float(ww), 1)); d.append(round(float(깊이), 1))
        토막.append("층쌓기(z=%s, x=%s, y=%s, w=%s, d=%s)" % (z, x, y, w, d))
    return 토막


def 켜재기(앞, 옆, fa, fo, 층수=60, 뺄=None):
    """켜마다 (z0, z1, y, 깊이, 앞 구간들) 아래→위. 뺄 = 앞 그림에서 빼고 잴 화소(혼합의 팔)."""
    rows = np.nonzero(앞.any(1))[0]
    경계 = np.linspace(rows.min(), rows.max() + 1, 층수 + 1)
    켜들 = []
    for r0, r1 in zip(경계[:-1], 경계[1:]):
        r = int((r0 + r1) / 2)
        z0, z1 = (fa.bot - r1) * fa.s, (fa.bot - r0) * fa.s
        옆값 = _옆줄(옆, fo, (z0 + z1) / 2)
        줄 = 앞[r] if 뺄 is None else 앞[r] & ~뺄[r]
        if 옆값 and 줄.any():
            켜들.append((z0, z1, 옆값[0], 옆값[1], _줄조각(줄)))
    return 켜들[::-1]


def 층타원매끈(앞, 옆, 키):
    fa, fo = 틀(앞, 키), 틀(옆, 키)
    return " + ".join(켜쌓기줄(켜재기(앞, 옆, fa, fo), fa)) or None


def 혼합매끈(앞, 옆, 키, 팔매듭=9):
    fa, fo = 틀(앞, 키), 틀(옆, 키)
    L = 부위나누기(앞)
    팔 = [번 for 번 in (3, 4) if (L == 번).sum() >= 30]
    if not 팔:
        return None
    토막 = 켜쌓기줄(켜재기(앞, 옆, fa, fo, 뺄=np.isin(L, 팔)), fa)
    for 번 in 팔:
        ys, xs = np.nonzero(L == 번)
        토막.append(_로프트줄(ys, xs, 번, 앞, 옆, fa, fo, 팔매듭))
    return " + ".join(토막)


def _겹켜재기(앞, 옆, fa, fo, 층수=60, 뺄=None):
    """켜마다 (z0, z1, [(a, b, y, 깊이)]) 아래→위. a, b = 정면 화소 구간 · y, 깊이 = **옆 구간마다** mm."""
    rows = np.nonzero(앞.any(1))[0]
    경계 = np.linspace(rows.min(), rows.max() + 1, 층수 + 1)
    켜들 = []
    for r0, r1 in zip(경계[:-1], 경계[1:]):
        r = int((r0 + r1) / 2)
        z0, z1 = (fa.bot - r1) * fa.s, (fa.bot - r0) * fa.s
        ro = int(np.clip(round(fo.bot - (z0 + z1) / 2 / fo.s - 0.5), 0, 옆.shape[0] - 1))
        상자 = [(a, b, -((c + d) / 2 - fo.c) * fo.s, (d - c) * fo.s) for a, b in _줄조각(앞[r] if 뺄 is None else 앞[r] & ~뺄[r]) for c, d in _줄조각(옆[ro])]
        if 상자:
            켜들.append((z0, z1, 상자))
    return 켜들[::-1]


def _겹(p, q):
    return min(p[1], q[1]) > max(p[0], q[0]) and abs(p[2] - q[2]) < (p[3] + q[3]) / 2


def 층겹침(앞, 옆, 키, 뺄=None):
    """층타원매끈 + 옆 그림의 빈 곳도 판다 (09-24 `벤치/겹침.py`: 안되던 17장 F@2mm 0.455 -> 0.509, 떨어진 것 0).
    켜마다 정면 구간 × 옆 구간마다 타원 — 두 그림 어느 쪽과도 어긋나지 않는다. 위아래는 x · y 둘 다 겹치고 1대1 이면 잇는다."""
    fa, fo = 틀(앞, 키), 틀(옆, 키)
    켜들 = _겹켜재기(앞, 옆, fa, fo, 뺄=뺄)
    줄기들, 열린 = [], []
    for i, (_, _, 상자) in enumerate(켜들):
        새 = []
        for g in 상자:
            닿 = [줄 for 줄 in 열린 if _겹(g, 줄[-1][1])]
            if len(닿) == 1 and sum(_겹(gg, 닿[0][-1][1]) for gg in 상자) == 1:
                닿[0].append((i, g))
                새.append(닿[0])
                continue
            줄기들.append([(i, g)])
            새.append(줄기들[-1])
        열린 = 새
    토막 = []
    for 줄 in 줄기들:
        z, x, y, w, d = [], [], [], [], []
        for n, (i, (a, b, yy, 깊이)) in enumerate(줄):
            z0, z1, _ = 켜들[i]
            for h in ([z0] if n == 0 else []) + [(z0 + z1) / 2] + ([z1 + 0.01] if n == len(줄) - 1 else []):
                z.append(round(float(h), 2)); x.append(round(float(((a + b) / 2 - fa.c) * fa.s), 1)); y.append(round(float(yy), 1))
                w.append(round(float((b - a) * fa.s), 1)); d.append(round(float(깊이), 1))
        토막.append("층쌓기(z=%s, x=%s, y=%s, w=%s, d=%s)" % (z, x, y, w, d))
    return " + ".join(토막) or None


def 혼합겹침(앞, 옆, 키, 팔매듭=9):
    """층겹침(몸통 · 다리) + 팔만 로프트 — 혼합 과 같은 팔. 09-24 exe 35장: 층겹침이 T 포즈 팔까지 켜로 쌓아 혼합이던 사람형 7장이 떨어졌다."""
    fa, fo = 틀(앞, 키), 틀(옆, 키)
    L = 부위나누기(앞)
    팔 = [번 for 번 in (3, 4) if (L == 번).sum() >= 30]
    if not 팔:
        return None
    몸 = 층겹침(앞, 옆, 키, 뺄=np.isin(L, 팔))
    토막 = [몸] if 몸 else []
    for 번 in 팔:
        ys, xs = np.nonzero(L == 번)
        토막.append(_로프트줄(ys, xs, 번, 앞, 옆, fa, fo, 팔매듭))
    return " + ".join(토막)


def 대보기(줄, 앞, 옆, 키):
    """후보 줄 하나 -> (앞 IoU, 옆 IoU). 입력 그림과 같은 틀에 그려 비교한다 — 정답은 안 쓴다."""
    V, F = D.실행(줄)
    fa, fo = 틀(앞, 키), 틀(옆, 키)
    return _iou(fa.그리기(V[:, 0], V[:, 2], F), 앞), _iou(fo.그리기(-V[:, 1], V[:, 2], F), 옆)


def 고르기(앞, 옆, 키):
    """-> 후보 목록(점수 높은 순) · 동점 수. 각 후보 = {이름, 줄, 앞, 옆, 점수}."""
    후보 = []
    for 이름, 줄 in 후보들(앞, 옆, 키):
        a, o = 대보기(줄, 앞, 옆, 키)
        후보.append({"이름": 이름, "줄": 줄, "앞": round(a, 4), "옆": round(o, 4), "점수": round((a + o) / 2, 4)})
    최고 = max(h["점수"] for h in 후보)
    묶음 = [h for h in 후보 if 최고 - h["점수"] <= 동점폭]               # 동점 — 그림 두 장이 못 가른 것들
    나머지 = sorted((h for h in 후보 if 최고 - h["점수"] > 동점폭), key=lambda h: -h["점수"])
    # 동점 안에서는 **카탈로그 순서**가 1순위다 (09-22 사용자 결정 「가」). 점수 순으로 두면 그리기 화소 오차가
    # 원뿔 ↔ 각뿔대를 정해 치마가 0.996 → 0.788 로 뒤집혔다. 앞에 둔 둥근 기본형이 피규어에 더 흔하다.
    return 묶음 + 나머지, len(묶음)


def 검사():
    """그림을 코드로 그려서 넣는다 — 앱이 원화3d 없이도 스스로 돈다."""
    def 판(그리기, W=400, H=400):
        im = Image.new("L", (W, H), 255)
        그리기(ImageDraw.Draw(im))
        return np.asarray(im) < 128
    원 = 판(lambda d: d.ellipse([100, 100, 300, 300], fill=0))
    네모 = 판(lambda d: d.rectangle([150, 50, 250, 350], fill=0))
    도넛 = 판(lambda d: (d.ellipse([100, 100, 300, 300], fill=0), d.ellipse([160, 160, 240, 240], fill=255)))
    납작 = 판(lambda d: d.rounded_rectangle([170, 100, 230, 300], radius=30, fill=0))      # 토러스 옆 = 폭 2r 인 경기장 꼴
    for 이름, 앞, 옆, 기대 in (("구", 원, 원, "구"), ("토러스", 도넛, 납작, "토러스")):
        후보, 동점 = 고르기(앞, 옆, 100.0)
        print("  %-6s 1순위 %-6s %.3f · 동점 %d · %s" % (이름, 후보[0]["이름"], 후보[0]["점수"], 동점, 후보[0]["줄"]))
        assert 후보[0]["이름"] == 기대 and 후보[0]["점수"] > 0.97, 후보[:2]      # 층 타원은 구멍도 켜마다 따라 그려 동점이 될 수 있다
    후보, 동점 = 고르기(네모, 네모, 100.0)
    print("  네모×네모 동점 %d — %s" % (동점, " · ".join(h["이름"] for h in 후보[:동점])))
    assert 동점 >= 3, "앞 · 옆이 네모면 상자 · 원기둥 · 45도 상자는 못 가른다"
    print("그림 검사 통과 — 구 · 토러스는 1순위로 고르고, 네모×네모는 동점으로 넘긴다")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "검사":
        검사()
    else:
        print(__doc__)
