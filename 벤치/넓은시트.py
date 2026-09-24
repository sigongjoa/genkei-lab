"""벤치 시트가 물체를 자른 장을 넓은 창으로 다시 굽는다 (09-24).

    python 벤치/넓은시트.py      -> out/시트넓게/<케이스>/{front,side}.png · 비교는 out/넓은시트.json

  원화3d 시트 창은 u ∈ [-0.6, 0.6] (키 1). 보물상자 Closed(반폭 0.79) · 메카 · 종 · Evil Wizard 는 잘려서
  입력이 이미 틀렸다. 서비스에선 사용자가 전체 그림을 준다 — 벤치도 그래야 공정하다.
  여기선 실루엣만 굽는다(exe 는 알파만 본다): 창 u ∈ [-0.9, 0.9] · z ∈ [-0.1, 1.1], 1536 × 1024. 투영은 win.silhouette 과 같다
  (앞 u = x · 옆 u = +y, 창 좌표). 원화3d 코드는 안 바꾼다.
  확인: 잘리지 않은 장을 이 시트로 돌려도 지표가 같아야 한다(대조 avatarsample_d).

예측 (09-24, 돌리기 전에 커밋):
  W1 잘린 4장 모두 F@2mm 겉이 오른다
  W2 대조 avatarsample_d 의 F@2mm 겉 차이 <= 0.01 (창 크기만 바뀌면 결과가 같다)
돌린 뒤 (v1, 1024² 에 u ±0.9) — 둘 다 틀림: 상자 0.272 -> 0.305 · 종 0.495 -> 0.561 은 올랐지만 메카 0.357 -> 0.345 ·
  Evil 0.574 -> 0.570, 대조 avatarsample_d 0.962 -> 0.906 (1순위 혼합 -> 층 겹침). 창만이 아니라 **배율**도 2/3 로 줄었다.
v2 예측 (배율은 원래 시트 그대로 1024/1.2 화소, 캔버스만 1536 × 1024 로 넓힘 · 돌리기 전에 커밋):
  W3 대조 avatarsample_d F@2mm 겉 차이 <= 0.01
  W4 잘린 4장 F@2mm 겉 중앙이 오른다
"""
import json
import os
import sys

import cv2
import numpy as np
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import 어댑터 as A           # noqa: E402
import 겉지표 as O           # noqa: E402
import 지표 as J             # noqa: E402
from exe벤치 import exe로, 시트   # noqa: E402

넓게 = os.path.join(A.OUT, "시트넓게")
잘린 = ["cloak_Chest_Closed_AngpV0HxD8", "mech_Mech_4UvIHxnoSR", "cloak_Cloche_Qz9XMYysBQ", "wizard_Evil_Wizard_bdxawstqtq"]
W, H, U0, U1, Z0, Z1 = 1536, 1024, -0.9, 0.9, -0.1, 1.1          # 원래 시트와 같은 배율(1024 / 1.2)


def 실루엣(V, F, 축):
    u, z = V[:, 축], V[:, 2]
    px = (u - U0) / (U1 - U0) * W
    py = (Z1 - z) / (Z1 - Z0) * H
    P = np.stack([px, py], 1).astype(np.int32)
    g = np.zeros((H, W), np.uint8)
    for tri in P[np.asarray(F)]:
        cv2.fillConvexPoly(g, tri, 1)
    return g.astype(bool)


def 굽기(case):
    t = trimesh.load(os.path.join(시트, case, "truth.glb"), force="mesh")
    d = os.path.join(넓게, case)
    os.makedirs(d, exist_ok=True)
    for 이름, 축 in (("front", 0), ("side", 1)):
        m = 실루엣(np.asarray(t.vertices), t.faces, 축)
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[m] = (0, 0, 0, 255)
        Image.fromarray(rgba).save(os.path.join(d, 이름 + ".png"))


def 재기(case, 폴더):
    m = trimesh.load(os.path.join(폴더, "메시.stl"), force="mesh")
    g = A.정답(case)
    답점, _ = O.겉점(g)
    r, _ = J.재기(J.mm메시(A.창(m.vertices), m.faces), J.mm메시(g["메시"].vertices, g["메시"].faces))
    f, 정, 재 = O.F겉(J.mm메시(A.창(m.vertices), m.faces), 답점)
    return {"F@2mm 겉": f, "F@2mm": r["F@2mm"], "챔퍼 중앙": r["챔퍼 중앙"], "새 각도 중앙": r["새 각도 중앙"],
            "1순위": json.load(open(os.path.join(폴더, "결과.json"), encoding="utf-8"))["후보"][0]["이름"]}


def main():
    기록 = {}
    for c in 잘린 + ["avatarsample_d"]:
        굽기(c)
        폴더 = os.path.join(A.OUT, "exe넓게", c)
        exe로(c, 폴더, 넓게)
        r = {"잘린 시트": 재기(c, os.path.join(A.OUT, "exe", c)), "넓은 시트": 재기(c, 폴더)}
        기록[c] = r
        print("%-32s 잘린 F겉 %.3f 챔퍼 %.2f (%s) -> 넓은 F겉 %.3f 챔퍼 %.2f (%s)" % (
            c[:32], r["잘린 시트"]["F@2mm 겉"], r["잘린 시트"]["챔퍼 중앙"], r["잘린 시트"]["1순위"],
            r["넓은 시트"]["F@2mm 겉"], r["넓은 시트"]["챔퍼 중앙"], r["넓은 시트"]["1순위"]), flush=True)
    json.dump(기록, open(os.path.join(A.OUT, "넓은시트.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
