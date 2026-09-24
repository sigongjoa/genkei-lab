"""자 여럿 — 3D IoU 하나로 「그림처럼 보이나」를 판정하지 않는다 (09-23, art2real `실험_20260923/지표.py` 를 옮김).

    python 벤치/지표.py        키트 벤치가 남긴 exe 메시(out/exe/<케이스>/메시.stl) -> out/지표.json · out/지표_*.png

  ① 3D IoU               지금까지 쓰던 것 (키트벤치.json 에서)
  ② 챔퍼 mm              우리 겉면 ↔ 정답 원본 메시 겉면 최근접 거리, 양방향 중앙 · 95%. 키 150 mm
  ③ F@2mm               겉면 점이 2 mm 안에 드는 몫(정밀 · 재현의 조화평균). 학계 표준 headline
  ④ 새 각도 실루엣 IoU    입력으로 쓴 앞 · 옆을 뺀 12 각도(비스듬 8 + 위에서 30° 4)에서 본 그림자
  ⑤ 표면각 중앙 · 90%      이웃 면 사이 각. 계단(층 타원 켜)은 90% 가 커진다
  ⑥ F@2mm 겉 (09-24 부터 대표 자)  정답의 **바깥 겉면**만으로 잰 F. 정답 메시는 부품을 겹쳐 박아 안 보이는 안쪽 면이 많아
                         (보물상자 겉면의 56 %) 그걸 재현율에 세면 어떤 복원도 못 맞힌다. `벤치/겉지표.py` 참고
정답은 복셀이 아니라 원본 메시(art2real: 복셀 정답은 45° 계단이라 표면을 못 잰다).

예측 (09-23, 돌리기 전에 커밋) — 사람형 35장:
  M1 새 각도 실루엣 중앙 >= 0.70, 그리고 앞 · 옆 실루엣 중앙보다 낮다
  M2 챔퍼 중앙의 중앙 <= 3 mm · 95% 의 중앙 <= 15 mm
  M3 3D IoU 와 챔퍼 중앙의 순위상관 <= -0.5 (두 자가 같은 쪽을 가리킨다)
  M4 표면각 90%: 우리 > 정답 (층 타원 켜의 계단)

돌린 뒤 (09-23, 사람형 35장) — 넷 다 맞음:
  새 각도 0.893 (앞옆 0.946) · 챔퍼 중앙 0.83 mm · 95% 8.14 mm · F@2mm 0.727 · IoU↔챔퍼 순위상관 -0.777
  표면각 90% 는 우리가 **35장 모두 정확히 90.0°** — 켜 계단이 표면을 지배한다(정답 중앙 83.5°, 30~101° 로 제각각).
  IoU 가 못 보던 것: cloak_Chest_Closed 는 IoU 0.754 인데 F@2mm 0.234 · 챔퍼 95% 24.8 mm (부피는 맞고 겉이 틀림).
  열지도에서 넘침은 허리(치마 · 벨트)와 팔 아래 — 옆 그림 깊이 겹침. 모자람은 손 · 발끝 · 칼 같은 가는 끝.
"""
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402

키 = 150.0
τ = 2.0
각도들 = [(y, 0) for y in (30, 60, 120, 150, 210, 240, 300, 330)] + [(y, 30) for y in (45, 135, 225, 315)]


def _돌림(yaw, pitch):
    Rz = trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 0, 1])[:3, :3]
    Rx = trimesh.transformations.rotation_matrix(np.radians(pitch), [1, 0, 0])[:3, :3]
    return Rz @ Rx


def 그림자(P, 가운데, 반, R, 크기=256):
    """겉면 점 -> 그림자 마스크. 우리와 정답에 같은 가운데 · 배율을 쓴다(각자 맞추면 크기 차이를 못 본다)."""
    q = ((P - 가운데) @ R.T)[:, [0, 2]]
    ij = np.rint(q * (크기 * 0.45 / 반) + 크기 / 2).astype(int)
    ij = ij[(ij >= 0).all(1) & (ij < 크기).all(1)]
    M = np.zeros((크기, 크기), bool)
    M[크기 - 1 - ij[:, 1], ij[:, 0]] = True
    M = ndimage.binary_closing(M, np.ones((3, 3), bool), iterations=2)
    return ndimage.binary_fill_holes(M)


def 표면각(m, n=40000):
    a = m.copy()
    a.merge_vertices()
    ang = np.degrees(a.face_adjacency_angles)
    if len(ang) > n:
        ang = np.random.default_rng(0).choice(ang, n, replace=False)
    return float(np.median(ang)), float(np.percentile(ang, 90))


def 재기(우, 답):
    """우 · 답 = mm 메시(같은 틀). -> 자 dict 와 그림용 (점, 거리, 그림자 쌍)."""
    pu, pd = 우.sample(60000, seed=0), 답.sample(60000, seed=1)
    d1, _ = cKDTree(pd).query(pu)
    d2, _ = cKDTree(pu).query(pd)
    d = np.concatenate([d1, d2])
    정밀, 재현 = float((d1 < τ).mean()), float((d2 < τ).mean())
    가운데 = (np.vstack([pu, pd]).max(0) + np.vstack([pu, pd]).min(0)) / 2
    반 = float(np.abs(np.vstack([pu, pd]) - 가운데).max())
    쌍, ious = [], []
    for yaw, pitch in 각도들:
        R = _돌림(yaw, pitch)
        a, b = 그림자(pu, 가운데, 반, R), 그림자(pd, 가운데, 반, R)
        ious.append(float((a & b).sum() / max((a | b).sum(), 1)))
        쌍.append((b, a))
    au, at = 표면각(우), 표면각(답)
    r = {"챔퍼 중앙": round(float(np.median(d)), 2), "챔퍼 95%": round(float(np.percentile(d, 95)), 2),
         "F@2mm": round(2 * 정밀 * 재현 / max(정밀 + 재현, 1e-9), 3), "정밀": round(정밀, 3), "재현": round(재현, 3),
         "새 각도 중앙": round(float(np.median(ious)), 3), "새 각도 최저": round(float(np.min(ious)), 3),
         "표면각 우리": [round(au[0], 2), round(au[1], 1)], "표면각 정답": [round(at[0], 2), round(at[1], 1)]}
    return r, (pu, d1, 쌍, ious)


def 겉점(g, n=60000):
    """정답 dict(어댑터.정답) -> (바깥 겉면 점 mm, 속면 몫). 층마다 채운 정답 부피를 2 칸 깎은 속에 든 점 = 안쪽 면."""
    from scipy import ndimage
    sys.path.insert(0, A.원화3d)
    import 정답 as JJ
    E = ndimage.binary_erosion(g["TF"], iterations=2)
    P = g["메시"].sample(n, seed=1)
    ij = np.rint(P / JJ.PITCH).astype(int) - g["ot"]
    ok = ((ij >= 0) & (ij < E.shape)).all(1)
    속 = np.zeros(len(P), bool)
    속[ok] = E[tuple(ij[ok].T)]
    return P[~속] * 키, float(속.mean())


def F겉(우mm, 답점):
    pu = 우mm.sample(60000, seed=0)
    d1, _ = cKDTree(답점).query(pu)
    d2, _ = cKDTree(pu).query(답점)
    정, 재 = float((d1 < τ).mean()), float((d2 < τ).mean())
    return round(2 * 정 * 재 / max(정 + 재, 1e-9), 3), round(정, 3), round(재, 3)


def mm메시(Vw, F):
    return trimesh.Trimesh(np.asarray(Vw) * 키, F, process=False)


def 열지도(pu, d1, 크기=300):
    """우리 겉면 점을 3/4 에서 찍고 정답까지 거리로 색칠 (파랑 0 · 빨강 >= 10 mm)."""
    R = _돌림(35, 18)
    q = (pu - pu.mean(0)) @ R.T
    o = np.argsort(q[:, 1])[::-1]                                        # 먼 것부터
    xy = q[o][:, [0, 2]]
    s = 크기 * 0.45 / np.abs(xy).max()
    t = np.clip(d1[o] / 10.0, 0, 1)
    c = (np.stack([t, 0.25 + 0 * t, 1 - t], 1) * 255).astype(np.uint8)
    im = Image.new("RGB", (크기, 크기), "white")
    px = im.load()
    for (x, z), cc in zip(xy, c):
        i, j = int(x * s + 크기 / 2), int(크기 / 2 - z * s)
        if 0 <= i < 크기 and 0 <= j < 크기:
            px[i, j] = tuple(cc)
    return im


def 판(c, r, 쌍, 열, iou3d):
    import 키트 as K
    im = Image.new("RGB", (1180, 420), "white")
    d = ImageDraw.Draw(im)
    d.text((14, 8), "%s — 3D IoU %.3f · 챔퍼 %.1f / %.1f mm · F@2mm %.3f · 새 각도 %.3f (최저 %.3f) · 표면각 %.1f° vs 정답 %.1f° (90%%)"
           % (c, iou3d, r["챔퍼 중앙"], r["챔퍼 95%"], r["F@2mm"], r["새 각도 중앙"], r["새 각도 최저"], r["표면각 우리"][1], r["표면각 정답"][1]),
           fill="black", font=K._글꼴(15))
    for k, (b, a) in enumerate(쌍[:8]):
        g = A.겹침(b, a)
        t = Image.fromarray(g).resize((130, 130))
        x, y = 14 + (k % 4) * 138, 40 + (k // 4) * 180
        im.paste(t, (x, y))
        d.text((x, y + 132), "%d° · %.2f" % (각도들[k][0], ((a & b).sum() / max((a | b).sum(), 1))), fill=(90, 90, 90), font=K._글꼴(13))
    im.paste(열, (580, 60))
    d.text((580, 370), "우리 겉면 → 정답 거리 (파랑 0 · 빨강 10 mm 이상)", fill=(90, 90, 90), font=K._글꼴(13))
    d.text((890, 60), "새 각도 그림자 겹침\n회색 = 둘 다\n빨강 = 정답만(모자람)\n파랑 = 우리만(넘침)", fill=(90, 90, 90), font=K._글꼴(14))
    return im


def main():
    기록벤치 = json.load(open(os.path.join(A.OUT, "키트벤치.json"), encoding="utf-8"))["기록"]
    기록 = {}
    for c, v in 기록벤치.items():
        if v["묶음"] == "도형":
            continue
        m = trimesh.load(os.path.join(A.OUT, "exe", c, "메시.stl"), force="mesh")
        답 = A.정답(c)["메시"]
        r, (pu, d1, 쌍, ious) = 재기(mm메시(A.창(m.vertices), m.faces), mm메시(답.vertices, 답.faces))
        g = A.정답(c)
        r["F@2mm 겉"], _, r["재현 겉"] = F겉(mm메시(A.창(m.vertices), m.faces), 겉점(g)[0])
        s = A.채점(m.vertices, m.faces, g)
        r.update({"3D IoU": v["exe"], "앞옆 실루엣": round((s["앞"] + s["옆"]) / 2, 3), "1순위": v["1순위"], "묶음": v["묶음"]})
        기록[c] = r
        판(c, r, 쌍, 열지도(pu, d1), v["exe"]).save(os.path.join(A.OUT, "지표_%s.png" % c))
        print("  %-34s IoU %.3f · 챔퍼 %4.1f/%5.1f · F %.3f · 새각도 %.3f (앞옆 %.3f) · 표면각90 %.1f/%.1f" % (
            c[:34], r["3D IoU"], r["챔퍼 중앙"], r["챔퍼 95%"], r["F@2mm"], r["새 각도 중앙"], r["앞옆 실루엣"],
            r["표면각 우리"][1], r["표면각 정답"][1]), flush=True)
    판정(기록)


def 판정(기록):
    from scipy.stats import spearmanr
    v = list(기록.values())
    med = lambda f: float(np.median([f(x) for x in v]))
    ρ = spearmanr([x["3D IoU"] for x in v], [x["챔퍼 중앙"] for x in v])[0]
    값 = [(med(lambda x: x["새 각도 중앙"]) >= 0.70 and med(lambda x: x["새 각도 중앙"]) < med(lambda x: x["앞옆 실루엣"]),
          "새 각도 %.3f · 앞옆 %.3f" % (med(lambda x: x["새 각도 중앙"]), med(lambda x: x["앞옆 실루엣"]))),
         (med(lambda x: x["챔퍼 중앙"]) <= 3 and med(lambda x: x["챔퍼 95%"]) <= 15,
          "챔퍼 중앙 %.2f mm · 95%% %.2f mm" % (med(lambda x: x["챔퍼 중앙"]), med(lambda x: x["챔퍼 95%"]))),
         (ρ <= -0.5, "순위상관 %.3f" % ρ),
         (med(lambda x: x["표면각 우리"][1]) > med(lambda x: x["표면각 정답"][1]),
          "표면각 90%% 우리 %.1f° · 정답 %.1f°" % (med(lambda x: x["표면각 우리"][1]), med(lambda x: x["표면각 정답"][1])))]
    예측 = [l.strip() for l in __doc__.splitlines() if l.strip().startswith("M")]
    print("\n예측")
    결과 = []
    for 말, (맞, 글) in zip(예측, 값):
        print("  %s %s\n      %s" % ("○" if 맞 else "✗", 말, 글))
        결과.append({"예측": 말, "맞음": bool(맞), "값": 글})
    print("  F@2mm 중앙 %.3f · F@2mm 겉 중앙 %.3f (대표)" % (med(lambda x: x["F@2mm"]), med(lambda x: x["F@2mm 겉"])))
    json.dump({"기록": 기록, "예측": 결과}, open(os.path.join(A.OUT, "지표.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
