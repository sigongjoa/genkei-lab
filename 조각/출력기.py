"""출력기 — 프리셋(레진 · FDM) · 글자 새김 · 공차 쿠폰 (09-25, 계획 `claudedocs/plan_기본형키트_20260925.md` 단계 0).

    python 조각/출력기.py 쿠폰 [프리셋 | 전부] [폴더]      -> 폴더/쿠폰_<프리셋>.stl · .png
    python 조각/출력기.py 검사

  사용자 실물 출력(09-25): 「연결부위 불량」. 공차는 출력기 · 재료 · 방향마다 다르다(보편 값 없음 — 원형사 교본 5장).
  그래서 **쿠폰**을 먼저 뽑아 알맞은 틈을 고르고, 그 값을 프리셋에 적는다.
  쿠폰 = 판 하나: 줄마다 다보 크기(폭 2 · 3 · 4 mm), 칸마다 한 변 틈(0.05 ~ 0.40 mm) 구멍 + 틈 숫자 새김 · 다보 3 개(손잡이 달린).
  다보는 키트와 같은 **사다리꼴**(한쪽이 좁다 — 한 자리만 맞는다) · 구멍 입구 모따기.
"""
import os
import sys

import numpy as np
import manifold3d as m3
from PIL import Image, ImageDraw, ImageFont
from skimage import measure

M = m3.Manifold

# 한 변 틈 · 새김 깊이 · 글자 높이 · 최소 두께 · 모따기 (mm). 시작값 — 쿠폰으로 확인해 고친다(원형사 교본 5 · 6장).
프리셋 = {
    "레진": {"틈": 0.10, "새김": 0.4, "글자": 3.0, "최소두께": 1.0, "모따기": 0.2},
    "FDM0.4": {"틈": 0.25, "새김": 0.6, "글자": 5.0, "최소두께": 1.2, "모따기": 0.4},
    "FDM0.2": {"틈": 0.15, "새김": 0.4, "글자": 3.5, "최소두께": 0.8, "모따기": 0.3},
}
쿠폰틈 = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
쿠폰크기 = [1.0, 1.5, 2.0]                                                 # 다보 반폭 r — 폭 2 · 3 · 4 mm


def _글꼴(px):
    for p in ("C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, px)
    return ImageFont.load_default()


def 글자(글, 높이, 깊이):
    """글 -> 바닥 z=0 에서 +z 로 깊이만큼 선 글자 입체(가운데 정렬 · 높이 mm). 새김은 이것을 빼면 된다.
    글꼴을 크게 그린 뒤 윤곽선 -> CrossSection(짝홀) -> 밀어냄. 결정적."""
    px = 160
    f = _글꼴(px)
    l, t, r, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), 글, font=f)
    im = Image.new("L", (r - l + 20, b - t + 20), 0)
    ImageDraw.Draw(im).text((10 - l, 10 - t), 글, fill=255, font=f)
    a = np.pad(np.asarray(im) > 127, 1).astype(float)
    s = 높이 / (b - t)
    polys = []
    for c in measure.find_contours(a, 0.5):
        polys.append(np.stack([(c[:, 1] - 1) * s, -(c[:, 0] - 1) * s], 1))
    cs = m3.CrossSection(polys, m3.FillRule.EvenOdd).simplify(0.01)
    g = M.extrude(cs, 깊이)
    lo, hi = np.asarray(g.bounding_box()[:3]), np.asarray(g.bounding_box()[3:])
    return g.translate([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, 0])


def 다보(r, 반, 여유=0.0, 모따기=0.0, 뒤집힘=False):
    """축 +z · 가운데 원점 · 길이 2반. 여유 > 0 이면 구멍(한 변 여유만큼 크고 0.3 mm 깊다) + 입구 모따기(위 끝)."""
    a, L = r + 여유, 반 + (0.3 if 여유 else 0.0)
    좁 = 0.5 * a
    단면 = [(-a, -a), (a, -a), (좁, a), (-좁, a)]
    if 뒤집힘:
        단면 = [(x, -y) for x, y in 단면]
    몸 = M.hull_points([[x, y, z] for z in (-L, L) for x, y in 단면])
    if 여유 and 모따기:
        k = (a + 모따기) / a
        몸 = 몸 + M.hull_points([[x, y, L - 모따기] for x, y in 단면] + [[x * k, y * k, L + 0.01] for x, y in 단면])
    return 몸


def 쿠폰(이름):
    p = 프리셋[이름]
    반 = 3.0                                                               # 다보 반길이 — 구멍 깊이 3.3 mm
    칸 = max(10.0, p["글자"] * 2.2)
    두께 = 반 + 0.3 + 1.5                                                    # 구멍 밑에 1.5 mm
    줄높이 = 칸 + p["글자"] + 2
    W = 칸 * len(쿠폰틈) + 14
    H = 줄높이 * len(쿠폰크기) + 6 + p["글자"] + 4
    판 = M.cube([W, H, 두께])
    구멍들, 글들 = [], []
    for i, r in enumerate(쿠폰크기):
        y0 = 6 + i * 줄높이 + 칸 / 2
        글들.append(글자("%d" % round(2 * r), p["글자"], p["새김"] + 0.01).translate([6, y0, 두께 - p["새김"]]))
        for j, g in enumerate(쿠폰틈):
            x0 = 14 + j * 칸 + 칸 / 2 - 2
            구멍들.append(다보(r, 반, g, p["모따기"]).translate([x0, y0, 두께 - (반 + 0.3)]))
            글들.append(글자("%02d" % round(g * 100), p["글자"] * 0.8, p["새김"] + 0.01)
                      .translate([x0, y0 + 칸 / 2 + p["글자"] * 0.3, 두께 - p["새김"]]))
    글들.append(글자(이름, p["글자"], p["새김"] + 0.01).translate([W / 2, H - p["글자"] / 2 - 2, 두께 - p["새김"]]))
    판 = 판 - M.batch_boolean(구멍들 + 글들, m3.OpType.Add)
    # 다보 3 개 — 손잡이 판(12 × 12 × 2 mm) 위에 서 있다 · 손잡이에 크기 새김
    못들 = []
    for i, r in enumerate(쿠폰크기):
        x = 6 + i * 18                                                      # 판 아래(앞)쪽에 — 출력판 폭을 줄인다
        손 = M.cube([12, 12, 2.0]).translate([x, -16, 0])
        못 = 다보(r, 반).translate([x + 6, -8, 2.0 + 반])
        못들.append(손 - 글자("%d" % round(2 * r), 3.0, 0.41).translate([x + 6, -13.5, 2.0 - 0.4]) + 못)
    return M.batch_boolean([판] + 못들, m3.OpType.Add)


def 그림(man, 경로, 제목=""):
    """3/4 면 그늘 그림(키트 렌더러)."""
    import 키트 as K
    me = man.to_mesh()
    V = np.asarray(me.vert_properties, np.float64)[:, :3]
    im = K.그리기({"쿠폰": (V, np.asarray(me.tri_verts))}, 크기=(900, 600), 색표={"쿠폰": (200, 200, 205)})
    ImageDraw.Draw(im).text((12, 10), 제목, fill="black", font=K._글꼴(18))
    im.save(경로)


def 검사():
    import trimesh
    g = 글자("L3", 5.0, 0.6)
    assert g.status() == m3.Error.NoError and g.volume() > 0
    for 이름 in 프리셋:
        c = 쿠폰(이름)
        me = c.to_mesh()
        t = trimesh.Trimesh(np.asarray(me.vert_properties)[:, :3], np.asarray(me.tri_verts), process=False)
        assert t.is_watertight and c.status() == m3.Error.NoError, 이름
        lo, hi = np.asarray(c.bounding_box()[:3]), np.asarray(c.bounding_box()[3:])
        print("  쿠폰 %-7s 닫힘 %s · 크기 %.0f × %.0f × %.1f mm · 정점 %d" % (이름, t.is_watertight, *(hi - lo), len(t.vertices)))
    # 다보가 제 구멍에 들어가고(틈 > 0), 90° 돌리면 안 들어간다
    못, 구 = 다보(1.5, 3.0), 다보(1.5, 3.0, 0.10, 0.2)
    assert (못 - 구).volume() < 1e-6, "제 구멍에 들어간다"
    assert (못.rotate([0, 0, 90]) - 구).volume() / 못.volume() > 0.05, "돌리면 안 들어간다"
    assert (못 - 다보(1.5, 3.0, 0.10, 0.2, 뒤집힘=True)).volume() / 못.volume() > 0.05, "뒤집힌 짝 구멍에 안 들어간다"
    print("출력기 검사 통과 — 글자 · 쿠폰 닫힘 · 다보 들어감/돌지 않음/짝 다름")


if __name__ == "__main__":
    if sys.argv[1] == "검사":
        검사()
    else:
        import trimesh
        이름들 = list(프리셋) if len(sys.argv) < 3 or sys.argv[2] == "전부" else [sys.argv[2]]
        폴더 = sys.argv[3] if len(sys.argv) > 3 else "."
        os.makedirs(폴더, exist_ok=True)
        for 이름 in 이름들:
            c = 쿠폰(이름)
            me = c.to_mesh()
            trimesh.Trimesh(np.asarray(me.vert_properties)[:, :3], np.asarray(me.tri_verts), process=False).export(
                os.path.join(폴더, "쿠폰_%s.stl" % 이름))
            그림(c, os.path.join(폴더, "쿠폰_%s.png" % 이름), "쿠폰 " + 이름)
            print("->", os.path.join(폴더, "쿠폰_%s.stl" % 이름))
