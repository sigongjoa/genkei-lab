"""그림 → 형태. 앞 · 옆 그림 두 장과 키(mm)만 보고 카탈로그에서 형태를 고른다. 정답은 모른다.

    python 조각/그림.py 검사

입력 규약: 흰(또는 투명) 바탕에 물체 하나. 앞 그림은 정면(오른쪽 = +x), 옆 그림은 **물체가 오른쪽을 본다**.
두 그림 다 물체 키를 `키` mm 로 잡는다 — 그래서 그림마다 해상도가 달라도 된다.

  1) 잰다      물체 범위 · 줄마다 폭 · 연결 성분 · 안쪽 구멍
  2) 후보      카탈로그 항목마다 잰 수치로 DSL 한 줄 (상자 · 원기둥 · 원뿔 · 구 · 45도 상자 · 토러스 · 두 덩어리)
  3) 대 본다   후보 메시를 **입력 그림과 같은 틀**에 그려 앞 · 옆 실루엣 IoU 평균 — 이게 앱이 아는 유일한 점수
  4) 고른다    가장 높은 것. 0.005 안의 후보는 **동점** — 그림 두 장으로는 못 가른다, 사람이 고를 몫
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
    for 축, m, f in (("y", 앞, fa), ("x", 옆, fo)):                 # 구멍이 보이는 판 = 토러스 축 방향
        hole = ndimage.binary_fill_holes(m) & ~m
        if hole.sum() > 50:
            hz = f.범위(hole)
            Ro, Ri = h / 2, (hz[3] - hz[2]) / 2
            out.append(("토러스", '토러스(R=%s, r=%s, 축="%s", %s)' % (r1((Ro + Ri) / 2), r1((Ro - Ri) / 2), 축, 놓기)))
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
    return out


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
    후보.sort(key=lambda h: -h["점수"])                         # 안정 정렬 — 동점이면 카탈로그 순서
    동점 = sum(1 for h in 후보 if 후보[0]["점수"] - h["점수"] <= 동점폭)
    return 후보, 동점


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
        assert 후보[0]["이름"] == 기대 and 동점 == 1 and 후보[0]["점수"] > 0.97, 후보[:2]
    후보, 동점 = 고르기(네모, 네모, 100.0)
    print("  네모×네모 동점 %d — %s" % (동점, " · ".join(h["이름"] for h in 후보[:동점])))
    assert 동점 >= 3, "앞 · 옆이 네모면 상자 · 원기둥 · 45도 상자는 못 가른다"
    print("그림 검사 통과 — 구 · 토러스는 혼자 고르고, 네모×네모는 동점으로 넘긴다")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "검사":
        검사()
    else:
        print(__doc__)
