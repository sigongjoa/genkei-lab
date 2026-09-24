"""35장 한눈에 — 케이스마다 입력 앞 · 옆 | 정답 | 우리(오차 색) | 우리 정면 (09-24).

    python 벤치/갤러리.py        -> out/갤러리_1.png … (나쁜 것부터 6장씩)
    python 벤치/갤러리.py 곡선   -> out/갤러리곡선_1.png … 입력 앞 | 정답 | 이전 exe(계단) | 곡선 | 곡선 정면 (곡선 F 나쁜 것부터)

  지금 exe 결과(out/exe) · 잘렸던 4장은 넓은 시트 결과(out/exe넓게). 자 = F@2mm 겉(대표) · 챔퍼 중앙.
  3/4 그림은 점 구름에 면 방향 그늘 — 정답은 회색, 우리는 정답 바깥 겉면까지 거리 색(파랑 0 · 노랑 2 mm · 빨강 6 mm 이상).
"""
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
import 지표 as J             # noqa: E402
import 키트 as K             # noqa: E402
from exe벤치 import 시트    # noqa: E402
from 넓은시트 import 잘린, 넓게  # noqa: E402

칸 = 230
빛 = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])


def 점그림(P, N, 색, yaw, pitch, 가운데, 반):
    R = J._돌림(yaw, pitch)
    q = (P - 가운데) @ R.T
    n = N @ R.T
    그늘 = 0.35 + 0.65 * np.clip(n @ (R @ 빛), 0, 1)
    o = np.argsort(q[:, 1])[::-1]
    s = 칸 * 0.45 / 반
    im = Image.new("RGB", (칸, 칸), "white")
    px = im.load()
    for (x, _, z), c, g in zip(q[o], 색[o], 그늘[o]):
        i, j = int(x * s + 칸 / 2), int(칸 / 2 - z * s)
        if 0 <= i < 칸 and 0 <= j < 칸:
            v = tuple(int(t * g) for t in c)
            for di in (0, 1):
                for dj in (0, 1):
                    if i + di < 칸 and j + dj < 칸:
                        px[i + di, j + dj] = v
    return im


def 거리색(d):
    t = np.clip(d / 6.0, 0, 1)[:, None]
    a, b, c = np.array([40, 90, 230]), np.array([240, 200, 40]), np.array([230, 40, 40])
    return np.where(t < 1 / 3, a + (b - a) * (t * 3), b + (c - b) * np.clip((t - 1 / 3) * 1.5, 0, 1))


def 입력(case):
    뿌리 = 넓게 if case in 잘린 else 시트
    out = []
    for n in ("front", "side"):
        im = Image.open(os.path.join(뿌리, case, n + ".png")).convert("RGBA")
        흰 = Image.new("RGBA", im.size, "white")
        im = Image.alpha_composite(흰, im).convert("RGB")
        a = np.asarray(im).mean(-1) < 250
        ys, xs = np.nonzero(a)
        im = im.crop((xs.min() - 4, ys.min() - 4, xs.max() + 5, ys.max() + 5))
        im.thumbnail((칸 - 10, 칸 - 10))
        t = Image.new("RGB", (칸, 칸), "white")
        t.paste(im, ((칸 - im.width) // 2, (칸 - im.height) // 2))
        out.append(t)
    return out


def 한줄(case, 기록):
    폴더 = os.path.join(A.OUT, "exe넓게" if case in 잘린 else "exe", case)
    m = trimesh.load(os.path.join(폴더, "메시.stl"), force="mesh")
    우 = J.mm메시(A.창(m.vertices), m.faces)
    g = A.정답(c := case)
    답점, 속 = J.겉점(g)
    답 = J.mm메시(g["메시"].vertices, g["메시"].faces)
    Pd, fd = 답.sample(30000, return_index=True, seed=2)
    Pu, fu = 우.sample(30000, return_index=True, seed=3)
    d, _ = cKDTree(답점).query(Pu)
    f, 정, 재 = J.F겉(우, 답점)
    r, _ = J.재기(우, 답)
    모두 = np.vstack([Pd, Pu])
    가운데, 반 = (모두.max(0) + 모두.min(0)) / 2, float(np.abs(모두 - (모두.max(0) + 모두.min(0)) / 2).max())
    회 = np.full((len(Pd), 3), 200.0)
    그림 = 입력(c) + [점그림(Pd, 답.face_normals[fd], 회, 35, 18, 가운데, 반),
                     점그림(Pu, 우.face_normals[fu], 거리색(d), 35, 18, 가운데, 반),
                     점그림(Pu, 우.face_normals[fu], 거리색(d), 0, 0, 가운데, 반)]
    이름 = json.load(open(os.path.join(폴더, "결과.json"), encoding="utf-8"))["후보"][0]["이름"]
    기록[c] = {"F@2mm 겉": f, "챔퍼 중앙": r["챔퍼 중앙"], "1순위": 이름, "속면": round(속, 2)}
    return 그림, "%s   F겉 %.3f · 챔퍼 %.2f mm · 새 각도 %.3f · 고른 후보 %s%s" % (
        c, f, r["챔퍼 중앙"], r["새 각도 중앙"], 이름, " · (넓은 시트)" if c in 잘린 else "")


def main():
    지금 = json.load(open(os.path.join(A.OUT, "겉지표.json"), encoding="utf-8"))
    넓 = json.load(open(os.path.join(A.OUT, "넓은시트.json"), encoding="utf-8"))
    점수 = {c: (넓[c]["넓은 시트"]["F@2mm 겉"] if c in 잘린 else v["F@2mm 겉"]) for c, v in 지금.items()}
    순서 = sorted(점수, key=점수.get)
    기록, 쪽 = {}, 6
    for p in range(0, len(순서), 쪽):
        묶 = 순서[p:p + 쪽]
        im = Image.new("RGB", (칸 * 5 + 20, 60 + len(묶) * (칸 + 34)), "white")
        d = ImageDraw.Draw(im)
        d.text((10, 8), "입력 앞 | 입력 옆 | 정답 (3/4) | 우리 (3/4, 정답까지 거리: 파랑 0 · 노랑 2 mm · 빨강 6 mm+) | 우리 (정면)",
               fill="black", font=K._글꼴(15))
        d.text((10, 30), "나쁜 것부터 · %d-%d / %d" % (p + 1, p + len(묶), len(순서)), fill=(110, 110, 110), font=K._글꼴(13))
        for k, c in enumerate(묶):
            그림, 글 = 한줄(c, 기록)
            y = 60 + k * (칸 + 34)
            d.text((10, y), 글, fill="black", font=K._글꼴(14))
            for i, g in enumerate(그림):
                im.paste(g, (10 + i * 칸, y + 22))
            print(글, flush=True)
        im.save(os.path.join(A.OUT, "갤러리_%d.png" % (p // 쪽 + 1)))
    json.dump(기록, open(os.path.join(A.OUT, "갤러리.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)




def 메시점(폴더, 답점):
    m = trimesh.load(os.path.join(폴더, "메시.stl"), force="mesh")
    우 = J.mm메시(A.창(m.vertices), m.faces)
    P, fi = 우.sample(30000, return_index=True, seed=3)
    d, _ = cKDTree(답점).query(P)
    return 우, P, 우.face_normals[fi], d


def 비교():
    곡 = json.load(open(os.path.join(A.OUT, "곡선.json"), encoding="utf-8"))
    순서 = sorted(곡, key=lambda c: 곡[c]["F@2mm 겉"])
    쪽 = 6
    for p in range(0, len(순서), 쪽):
        묶 = 순서[p:p + 쪽]
        im = Image.new("RGB", (칸 * 5 + 20, 60 + len(묶) * (칸 + 34)), "white")
        d = ImageDraw.Draw(im)
        d.text((10, 8), "입력 앞 | 정답 (3/4) | 이전 exe — 계단 (3/4) | 곡선 — 부드럽게 합침 (3/4) | 곡선 (정면)   색: 정답까지 파랑 0 · 노랑 2 · 빨강 6 mm+",
               fill="black", font=K._글꼴(15))
        d.text((10, 30), "곡선 F 나쁜 것부터 · %d-%d / %d" % (p + 1, p + len(묶), len(순서)), fill=(110, 110, 110), font=K._글꼴(13))
        for k, c in enumerate(묶):
            g = A.정답(c)
            답점, _ = J.겉점(g)
            답 = J.mm메시(g["메시"].vertices, g["메시"].faces)
            Pd, fd = 답.sample(30000, return_index=True, seed=2)
            이전 = 메시점(os.path.join(A.OUT, "exe넓게" if c in 잘린 else "exe", c), 답점)
            곡선 = 메시점(os.path.join(A.OUT, "곡선", c), 답점)
            모두 = np.vstack([Pd, 이전[1], 곡선[1]])
            가운데 = (모두.max(0) + 모두.min(0)) / 2
            반 = float(np.abs(모두 - 가운데).max())
            그림 = [입력(c)[0], 점그림(Pd, 답.face_normals[fd], np.full((len(Pd), 3), 200.0), 35, 18, 가운데, 반),
                  점그림(이전[1], 이전[2], 거리색(이전[3]), 35, 18, 가운데, 반),
                  점그림(곡선[1], 곡선[2], 거리색(곡선[3]), 35, 18, 가운데, 반),
                  점그림(곡선[1], 곡선[2], 거리색(곡선[3]), 0, 0, 가운데, 반)]
            f0, _, _ = J.F겉(이전[0], 답점)
            v = 곡[c]
            y = 60 + k * (칸 + 34)
            d.text((10, y), "%s   F겉 이전 %.3f -> 곡선 %.3f · 챔퍼 %.2f mm · 표면각 90%% %.0f° (정답 %.0f°) · %s" % (
                c, f0, v["F@2mm 겉"], v["챔퍼 중앙"], v["표면각 우리"][1], v["표면각 정답"][1], v["1순위"]), fill="black", font=K._글꼴(14))
            for i, t in enumerate(그림):
                im.paste(t, (10 + i * 칸, y + 22))
            print(c, flush=True)
        im.save(os.path.join(A.OUT, "갤러리곡선_%d.png" % (p // 쪽 + 1)))


if __name__ == "__main__":
    비교() if len(sys.argv) > 1 and sys.argv[1] == "곡선" else main()
