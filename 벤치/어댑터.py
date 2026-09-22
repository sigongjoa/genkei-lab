"""조각 메시(mm) → 원화3d 벤치 창 좌표 → 채점.

    python 벤치/어댑터.py 검사     정답을 조각 좌표로 보냈다 돌아와 IoU ≈ 1 · 틀린 축은 떨어진다 · 그림

좌표 규약 (09-22 잰 것):
  조각 앱   mm · z 위 · 카메라가 -y 에서 본다 → 캐릭터 정면(얼굴 · 발끝)은 -y, 화면 오른쪽은 +x
  벤치 창   키 1 · z 위 · 바닥 z=0 · x·y 가운데. 정면(얼굴 · 발끝)은 +y, front.png 오른쪽은 +x
            (avatarsample_d 발 y 평균 +0.031 > 정강이 -0.019 · front.png = win.silhouette 그대로 IoU 1.000)
  그래서 y 만 뒤집고 win.normalize 로 앉힌다. 정합(ICP)은 안 한다 — 좌표가 맞아야 점수가 맞다.
"""
import os
import sys

import numpy as np
import trimesh
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
원화3d = os.path.normpath(os.path.join(HERE, "..", "..", "원화3d"))    # art2real 안쪽에 있을 때만 (공개 저장소엔 없다)
sys.path.insert(0, 원화3d)
import 정답 as J      # noqa: E402  grid · common · PITCH
import win as W       # noqa: E402  normalize · silhouette

OUT = os.path.join(HERE, "out")
뒤집기 = np.array([1.0, -1.0, 1.0])                 # 조각 → 창: 정면 -y → +y


def 창(V):
    """조각 정점(mm) → 벤치 창 좌표."""
    return W.normalize(np.asarray(V, np.float64) * 뒤집기)


def 조각좌표(Vw, 키=150.0):
    """거꾸로: 창 좌표 → 조각 앱이 `열기` 로 여는 것과 같은 모양(가운데 · 키 150 mm). 검사용."""
    V = np.asarray(Vw, np.float64) * 뒤집기
    return (V - (V.max(0) + V.min(0)) / 2) * (키 / max(np.ptp(V[:, 2]), 1e-9))


def fill_slices(T):
    """z 층마다 닫힌 구멍을 채운다 — `원화3d/실험_20260915/겉면채점.fill_slices` 와 같다 (그쪽은 import 하면 chdir 한다)."""
    out = T.copy()
    for k in range(T.shape[2]):
        if T[:, :, k].any():
            out[:, :, k] = ndimage.binary_fill_holes(T[:, :, k])
    return out


def 정답(case):
    t = trimesh.load(os.path.join(원화3d, "out", "시트", case, "truth.glb"), force="mesh")
    T, ot = J.grid(t, J.PITCH)
    return {"메시": t, "TF": fill_slices(T), "ot": ot}


def 실루엣(Vw, F, view):
    return np.asarray(W.silhouette(Vw, np.asarray(F), view)).astype(bool)


def iou(a, b):
    return round(float((a & b).sum() / max((a | b).sum(), 1)), 5)


def 채점(V, F, 답):
    """조각 메시 하나를 정답 하나에 잰다. 실루엣은 1024² 앞 · 옆.
    IoU3D      양쪽 다 z 층마다 속을 채운다 — 굿즈 속은 차도 된다(09-15). 조각 쪽 기본 자.
    IoU3D_점수판 정답만 채운다 — 원화3d 점수판(`겉면채점.bench_one`) 과 같은 값. 속 빈 메시(치마)는 여기서 깎인다."""
    Vw = 창(V)
    G, og = J.grid(trimesh.Trimesh(Vw, F, process=False), J.PITCH)
    A, B = J.common(답["TF"], 답["ot"], G, og)
    A2, B2 = J.common(답["TF"], 답["ot"], fill_slices(G), og)
    tV, tF = np.asarray(답["메시"].vertices), np.asarray(답["메시"].faces)
    return {"IoU3D": iou(A2, B2), "IoU3D_점수판": iou(A, B),
            "앞": iou(실루엣(tV, tF, "front"), 실루엣(Vw, F, "front")),
            "옆": iou(실루엣(tV, tF, "side"), 실루엣(Vw, F, "side"))}


# ── 검사 ─────────────────────────────────────────────────────────────────────

도형 = ["도형_" + n for n in ("상자", "원기둥", "원뿔", "구", "두기둥대각", "두기둥나란히", "상자45도", "L자", "치마", "토러스세움", "토러스옆")]


def 겹침(t, m):
    """둘 다 = 회색 · 정답만 = 빨강 · 조각만 = 파랑."""
    g = np.full(t.shape + (3,), 255, np.uint8)
    g[t & m] = (150, 150, 150); g[t & ~m] = (255, 59, 48); g[~t & m] = (0, 122, 255)
    return g


def 그림(칸들, 머리, 부제, 이름, 열=4, S=150):
    """칸 = (제목, [(정답 마스크, 조각 마스크), ...], 점수 글, 빨간 제목?) -> out/<이름>."""
    from PIL import Image, ImageDraw
    from 계획 import _font                                      # 원화3d 의 한글 글꼴
    글 = _font(15)
    n = max(len(c[1]) for c in 칸들)
    w = n * (S + 4) + 16
    행 = (len(칸들) + 열 - 1) // 열
    im = Image.new("RGB", (열 * w + 20, 행 * (S + 58) + 70), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 14), 머리, fill="black", font=_font(18))
    d.text((20, 40), 부제, fill=(110, 110, 110), font=글)
    for i, (제목, 쌍들, 점수, 빨강) in enumerate(칸들):
        x, y = 20 + (i % 열) * w, 70 + (i // 열) * (S + 58)
        for j, (t, m) in enumerate(쌍들):
            ys, xs = np.nonzero(t | m)
            box = (max(xs.min() - 8, 0), max(ys.min() - 8, 0), xs.max() + 8, ys.max() + 8)
            tile = Image.fromarray(겹침(t, m)).crop(box)
            tile.thumbnail((S, S))
            im.paste(tile, (x + j * (S + 4) + (S - tile.width) // 2, y + (S - tile.height) // 2))
        d.text((x, y + S + 4), 제목, fill=(255, 59, 48) if 빨강 else "black", font=글)
        d.text((x, y + S + 24), 점수, fill=(110, 110, 110), font=글)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 이름)
    im.save(p)
    return p


def 검사():
    sys.path.insert(0, os.path.join(HERE, "..", "조각"))
    from engine import 조각
    칸들 = []                                                   # (제목, 정답 앞, 조각 앞, 정답 옆, 조각 옆, 점수)
    def 한칸(제목, case, V, F, 답=None):
        답 = 답 or 정답(case)
        s = 채점(V, F, 답)
        tV, tF, Vw = np.asarray(답["메시"].vertices), 답["메시"].faces, 창(V)
        칸들.append((제목, 실루엣(tV, tF, "front"), 실루엣(Vw, F, "front"), 실루엣(tV, tF, "side"), 실루엣(Vw, F, "side"), s))
        print("  %-22s IoU3D %.4f (점수판 %.4f) · 앞 %.4f · 옆 %.4f" % (제목, s["IoU3D"], s["IoU3D_점수판"], s["앞"], s["옆"]))
        return s, 답

    print("1) 왕복: 정답 → 조각 좌표(mm, 키 150) → 조각 엔진 → 어댑터 → 정답과 비교. 1 이어야 한다")
    for c in 도형 + ["avatarsample_d"]:
        t = trimesh.load(os.path.join(원화3d, "out", "시트", c, "truth.glb"), force="mesh")
        e = 조각(조각좌표(t.vertices), t.faces)                   # 조각 앱이 들고 있는 그대로의 정점
        assert abs(np.ptp(e.V[:, 2]) - 150) < 1e-6
        s, 답 = 한칸(c.replace("도형_", ""), c, e.V, e.F)
        assert s["IoU3D"] > 0.999 and s["앞"] > 0.999 and s["옆"] > 0.999, (c, s)

    print("2) 정면 방향: 조각 좌표에서 발끝이 카메라(-y) 쪽이어야 한다")
    V = 조각좌표(답["메시"].vertices)                              # avatarsample_d
    발, 정강이 = V[V[:, 2] < V[:, 2].min() + 3], V[(V[:, 2] > V[:, 2].min() + 15) & (V[:, 2] < V[:, 2].min() + 30)]
    print("  발 y 평균 %.1f mm · 정강이 y 평균 %.1f mm" % (발[:, 1].mean(), 정강이[:, 1].mean()))
    assert 발[:, 1].mean() < 정강이[:, 1].mean()

    print("3) 질 수 있어야 한다: 축을 틀리면 · 다른 모양이면 떨어진다")
    틀림 = 조각좌표(답["메시"].vertices) * 뒤집기                  # y 를 안 뒤집은 셈
    s1, _ = 한칸("avatar · y 안 뒤집음", "avatarsample_d", 틀림, 답["메시"].faces, 답)
    L = trimesh.load(os.path.join(원화3d, "out", "시트", "도형_L자", "truth.glb"), force="mesh")
    s2, _ = 한칸("L자 · 앞뒤 뒤집힘", "도형_L자", 조각좌표(L.vertices) * 뒤집기, L.faces)
    구 = trimesh.creation.icosphere(4, 50.0)
    s3, _ = 한칸("avatar · 구 그대로", "avatarsample_d", 구.vertices, 구.faces, 답)
    assert s1["IoU3D"] < 0.9 and s1["옆"] < 0.9, s1
    assert s2["IoU3D"] < 0.9, s2
    assert s3["IoU3D"] < 0.5, s3

    p = 그림([(c[0], [(c[1], c[2]), (c[3], c[4])], "3D %.3f (판 %.3f) · 앞 %.3f · 옆 %.3f" % (c[5]["IoU3D"], c[5]["IoU3D_점수판"], c[5]["앞"], c[5]["옆"]),
               c[5]["IoU3D"] < 0.9) for c in 칸들],
             "벤치 어댑터 검사 — 회색 겹침 · 빨강 정답만 · 파랑 조각만 (앞 | 옆)",
             "위 12 칸: 왕복 (1 이어야 함) · 마지막 3 칸: 일부러 틀린 것 (떨어져야 함)", "어댑터_검사.png")
    print("검사 통과 — 왕복 %d 장 전부 > 0.999 · 정면 -y · 틀린 셋은 떨어짐 ->" % (len(칸들) - 3), p)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "검사":
        검사()
    else:
        print(__doc__)
