"""켜로 자르지 않는다 — 앞 · 옆 그림자에서 바로 곡면 (09-24 사용자: 「갑자기 왜 오브젝트가 슬라이스 되는건데?」).

    python 벤치/그림자곡면.py [둥글기mm]     -> out/그림자곡면.png · out/그림자곡면.json

  켜(수평 60 층) 방식은 누운 부위(모자 챙 · T 포즈 팔)를 얇은 원반 여러 장으로 쌓았다 — 그게 「슬라이스」.
  여기선 방향이 없다: 앞 그림 경계까지 2D 거리 d앞(x, z) · 옆 그림 경계까지 d옆(y, z) (속 -)
  -> 둥근 교집합 f = |max(d앞 + r, d옆 + r, 0)| + min(max(d앞 + r, d옆 + r), 0) - r   (r = 둥글기)
  -> 두 그림자가 만나는 모서리가 어느 방향이든 r 로 둥글다. 옆 그림의 빈 곳(겹침)도 저절로 파인다.
  곡면은 manifold level_set. 비교: 지금 곡선(out/곡선) · 정답. 자 = F@2mm 겉 · 챔퍼 + 그림.

돌린 뒤 (09-24, 6장):
  ① 그림자 곡면(둥근 교집합, r 4 mm) — 실패: 단면이 둥근 네모(윤곽 교집합)라 뭉툭 · F겉 0.1~0.3 떨어짐(로봇 0.782 -> 0.472).
  ② 켜 부피(`python 벤치/그림자곡면.py 0`) — 줄기 없이 높이 1 mm 마다 (앞 구간 × 옆 구간) 타원을 칸에 칠하고 거리장을 z 로도 흐림:
     F겉 지금 곡선과 같은 급(마녀 0.783 -> 0.784 · 노움 0.721 -> 0.738 · 기사 0.850 -> 0.853 · 로봇 0.782 -> 0.777 ·
     avatarsample_f 0.930 -> 0.905 · 다리벌림 0.892 -> 0.902). 끊긴 줄기의 뚜껑 원반(팔 밑 판 · 이음매 턱)이 사라졌다.
     남는 것: 기운 모자 챙 — 두 그림 다 가는 띠라 높이마다 타원 = 계단 원판. 두 장의 한계 -> 모자 카탈로그 항목 몫.
"""
import json
import os
import sys

import numpy as np
import trimesh
import manifold3d as m3
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
import 지표 as J             # noqa: E402
import 그림 as G             # noqa: E402
import dsl as D              # noqa: E402
import 키트 as K             # noqa: E402
from exe벤치 import 시트    # noqa: E402
from 넓은시트 import 잘린, 넓게  # noqa: E402

케이스 = ["witch_Witch_vWI9PHfjcy", "wizard_WIzard_Gnome_dEuyzEgrF4", "knight_Warrior_Z6ZUtm6kc1", "robot_Robot_ejDr8lRglP",
        "avatarsample_f", "마네킹_다리벌림"]


def _거리2d(m, s):
    """마스크 -> 부호 거리(mm, 속 -)."""
    return (ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)) * s


def 곡면(앞, 옆, 키=150.0, r=4.0, 격자=1.0):
    fa, fo = G.틀(앞, 키), G.틀(옆, 키)
    da, do = _거리2d(앞, fa.s), _거리2d(옆, fo.s)
    x0, x1, z0, z1 = fa.범위()
    y0, y1 = fo.범위()[:2]
    lo = np.array([x0, -y1, 0.0]) - 2 * 격자
    hi = np.array([x1, -y0, z1]) + 2 * 격자
    n = np.ceil((hi - lo) / 격자).astype(int) + 1
    X = lo[0] + np.arange(n[0]) * 격자
    Y = lo[1] + np.arange(n[1]) * 격자
    Z = lo[2] + np.arange(n[2]) * 격자
    # 앞 그림 화소 (열 = x, 행 = z) · 옆 그림 화소 (열 = -y: 옆 그림 오른쪽 = 정면 = -y)
    ca = np.clip(np.rint(fa.c + X / fa.s - 0.5).astype(int), 0, 앞.shape[1] - 1)
    ra = np.clip(np.rint(fa.bot - Z / fa.s - 0.5).astype(int), 0, 앞.shape[0] - 1)
    co = np.clip(np.rint(fo.c - Y / fo.s - 0.5).astype(int), 0, 옆.shape[1] - 1)
    ro = np.clip(np.rint(fo.bot - Z / fo.s - 0.5).astype(int), 0, 옆.shape[0] - 1)
    A2 = da[np.ix_(ra, ca)].T                                          # [x, z]
    O2 = do[np.ix_(ro, co)].T                                          # [y, z]
    a = A2[:, None, :] + r
    o = O2[None, :, :] + r
    f = np.sqrt(np.maximum(a, 0) ** 2 + np.maximum(o, 0) ** 2) + np.minimum(np.maximum(a, o), 0) - r
    f = ndimage.gaussian_filter(f, 0.7)
    f[[0, -1]] = f[:, [0, -1]] = f[:, :, [0, -1]] = 5.0
    nx, ny, nz = f.shape

    def 값(x, y, z):
        u, v, w = (x - lo[0]) / 격자, (y - lo[1]) / 격자, (z - lo[2]) / 격자
        i, j, k = min(max(int(u), 0), nx - 2), min(max(int(v), 0), ny - 2), min(max(int(w), 0), nz - 2)
        fu, fv, fw = min(max(u - i, 0.0), 1.0), min(max(v - j, 0.0), 1.0), min(max(w - k, 0.0), 1.0)
        c = f[i:i + 2, j:j + 2, k:k + 2]
        c = c[0] * (1 - fu) + c[1] * fu
        c = c[0] * (1 - fv) + c[1] * fv
        return -float(c[0] * (1 - fw) + c[1] * fw)

    man = m3.Manifold.level_set(값, list(lo) + list(lo + (n - 1) * 격자), 격자)
    return D.메시(man)


def 켜부피(앞, 옆, 키=150.0, 격자=1.0, 흐림=1.2):
    """줄기 없이 — 높이 격자마다 그 높이 앞 · 옆 줄의 구간 쌍(정면 구간 × 옆 구간)마다 타원을 칸에 칠하고,
    거리장을 (z 로도) 흐려 층을 잇는다. 끊긴 줄기의 뚜껑 원반이 없다."""
    fa, fo = G.틀(앞, 키), G.틀(옆, 키)
    x0, x1, z0, z1 = fa.범위()
    y0, y1 = fo.범위()[:2]
    lo = np.array([x0, -y1, 0.0]) - 4 * 격자
    hi = np.array([x1, -y0, z1]) + 4 * 격자
    n = np.ceil((hi - lo) / 격자).astype(int) + 1
    X = lo[0] + np.arange(n[0]) * 격자
    Y = lo[1] + np.arange(n[1]) * 격자
    속 = np.zeros(n, bool)
    for k in range(n[2]):
        z = lo[2] + k * 격자
        if not (0 <= z <= z1):
            continue
        ra = int(np.clip(round(fa.bot - z / fa.s - 0.5), 0, 앞.shape[0] - 1))
        ro = int(np.clip(round(fo.bot - z / fo.s - 0.5), 0, 옆.shape[0] - 1))
        for a, b in G._줄조각(앞[ra]):
            cx, w = ((a + b) / 2 - fa.c) * fa.s, (b - a) * fa.s
            for c_, d_ in G._줄조각(옆[ro]):
                cy, dd = -((c_ + d_) / 2 - fo.c) * fo.s, (d_ - c_) * fo.s
                속[:, :, k] |= ((X[:, None] - cx) / max(w / 2, 1e-3)) ** 2 + ((Y[None, :] - cy) / max(dd / 2, 1e-3)) ** 2 <= 1
    f = (ndimage.distance_transform_edt(~속) - ndimage.distance_transform_edt(속)) * 격자
    f = ndimage.gaussian_filter(f, 흐림)
    f[[0, -1]] = f[:, [0, -1]] = f[:, :, [0, -1]] = 5.0
    return _면(f, lo, n, 격자)


def _면(f, lo, n, 격자):
    nx, ny, nz = f.shape

    def 값(x, y, z):
        u, v, w = (x - lo[0]) / 격자, (y - lo[1]) / 격자, (z - lo[2]) / 격자
        i, j, k = min(max(int(u), 0), nx - 2), min(max(int(v), 0), ny - 2), min(max(int(w), 0), nz - 2)
        fu, fv, fw = min(max(u - i, 0.0), 1.0), min(max(v - j, 0.0), 1.0), min(max(w - k, 0.0), 1.0)
        c = f[i:i + 2, j:j + 2, k:k + 2]
        c = c[0] * (1 - fu) + c[1] * fu
        c = c[0] * (1 - fv) + c[1] * fv
        return -float(c[0] * (1 - fw) + c[1] * fw)

    return D.메시(m3.Manifold.level_set(값, list(lo) + list(lo + (n - 1) * 격자), 격자))


def main(r):
    크기 = (300, 380)
    im = Image.new("RGB", (크기[0] * 3 + 20, 40 + len(케이스) * (크기[1] + 30)), "white")
    d = ImageDraw.Draw(im)
    d.text((10, 8), "정답 | 지금 곡선 (줄기 + 부드럽게) | %s — 같은 배율 · 3/4" % ("켜 부피 (줄기 없이 타원을 칸에 칠함)" if r == 0 else "그림자 곡면 (둥글기 %.0f mm)" % r), fill="black", font=K._글꼴(16))
    기록 = {}
    for k, c in enumerate(케이스):
        뿌리 = 넓게 if c in 잘린 else 시트
        앞, 옆 = (G.마스크(os.path.join(뿌리, c, n + ".png")) for n in ("front", "side"))
        V, F = 켜부피(앞, 옆) if r == 0 else 곡면(앞, 옆, r=r)
        g = A.정답(c)
        답점, _ = J.겉점(g)
        새 = J.mm메시(A.창(V), F)
        곡 = trimesh.load(os.path.join(A.OUT, "곡선", c, "메시.stl"), force="mesh")
        지금 = J.mm메시(A.창(곡.vertices), 곡.faces)
        f새, f지금 = J.F겉(새, 답점)[0], J.F겉(지금, 답점)[0]
        기록[c] = {"F@2mm 겉 지금 곡선": f지금, "F@2mm 겉 그림자 곡면": f새, "닫힘": bool(trimesh.Trimesh(V, F, process=False).is_watertight)}
        print("%-32s 지금 %.3f -> 그림자 곡면 %.3f · 닫힘 %s" % (c[:32], f지금, f새, 기록[c]["닫힘"]), flush=True)
        뒤 = np.array([1.0, -1.0, 1.0])
        셋 = [(np.asarray(g["메시"].vertices) * 150 * 뒤, np.asarray(g["메시"].faces)[:, ::-1]),
              (np.asarray(지금.vertices) * 뒤, np.asarray(지금.faces)[:, ::-1]), (np.asarray(새.vertices) * 뒤, np.asarray(새.faces)[:, ::-1])]
        R = K._돌림()
        xy = np.vstack([(V_ @ R.T)[:, [0, 2]] for V_, _ in 셋])
        y = 40 + k * (크기[1] + 30)
        d.text((10, y), "%s   F겉 지금 %.3f · 그림자 곡면 %.3f" % (c, f지금, f새), fill="black", font=K._글꼴(14))
        for i, ((V_, F_), 색) in enumerate(zip(셋, [(200, 200, 200), (150, 185, 215), (160, 205, 160)])):
            im.paste(K.그리기({"몸": (V_, F_)}, 크기=크기, 색표={"몸": 색}, 범위=(xy.min(0), xy.max(0))), (10 + i * 크기[0], y + 20))
    im.save(os.path.join(A.OUT, "켜부피.png" if r == 0 else "그림자곡면.png"))
    json.dump(기록, open(os.path.join(A.OUT, "그림자곡면.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0)                 # 0 = 켜부피
