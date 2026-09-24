"""F@2mm 를 정답의 **바깥 겉면**으로만 (09-24).

    python 벤치/겉지표.py      -> out/겉지표.json

  정답 메시(게임 · VRM)는 부품을 겹쳐 박아 만들어 안쪽 면이 많다. 층마다 채운 정답 부피(A.정답 TF)를 2 칸 깎은 속에
  든 정답 겉면 점 = 밖에서 안 보이는 면. 이것까지 재현율에 세면 어떤 복원도 못 맞힌다(보물상자 겉면의 56 %).
  우리 메시는 manifold 합집합이라 안쪽 면이 없다 — 정답 쪽만 거른다. 거른 몫 = 「속면」.
  마네킹(기본형 겹친 것)도 0.09~0.19 — 합집합이 아닌 정답은 다 있다.

잰 것 (09-24, 오늘 exe 35장) — 「속 빔」 대책을 만들기 전에 원인부터 쟀다:
  35장 F@2mm 중앙 0.738 -> 겉만 0.790. 치마 0.679 -> 0.845 · 치비 마녀 0.685 -> 0.789 · 로봇(다리 총) 0.539 -> 0.660.
  속이 비어서(밖에서 들여다보이는 빈 통) 안 되는 장은 없었다: 모자 속면 0.00 · 마법사 0.06 · statue 0.09 · 종 0.17.
  아직 낮은 것(겉 F < 0.6)의 원인 둘:
    ① 입력이 잘렸다 — 물체가 시트 창(±0.6)보다 넓다: 보물상자 Closed(반폭 0.79) · 메카 · 종 · Evil Wizard. 벤치 시트 문제.
    ② 실루엣으로 안 보이는 오목 — 마법사(둥근 덩어리, 12 각도 그림자 0.86~0.92 인데 겉면 10 mm 넘게 어긋남) · 모자 ·
       statue(부조) · 상자 O72. 정면으로 파인 곳은 어느 그림자에도 안 나온다.
  -> 껍질 기본형은 안 만든다(고칠 병이 없다).
"""
import json
import os
import sys

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import 어댑터 as A           # noqa: E402
import 지표 as J             # noqa: E402
sys.path.insert(0, A.원화3d)
import 정답 as JJ            # noqa: E402


겉점, F겉 = J.겉점, J.F겉                                       # 09-24 지표.py 로 옮김


def main():
    전 = json.load(open(os.path.join(A.OUT, "지표_0923.json"), encoding="utf-8"))["기록"]
    지금 = json.load(open(os.path.join(A.OUT, "지표_0924.json"), encoding="utf-8"))["기록"]
    기록 = {}
    for c in 전:
        g = A.정답(c)
        답점, 속 = 겉점(g)
        m = trimesh.load(os.path.join(A.OUT, "exe", c, "메시.stl"), force="mesh")
        f, 정, 재 = F겉(J.mm메시(A.창(m.vertices), m.faces), 답점)
        기록[c] = {"속면": round(속, 2), "F@2mm": 지금[c]["F@2mm"], "F@2mm 겉": f, "정밀": 정, "재현 겉": 재}
        print("%-32s 속면 %.2f · F %.3f -> 겉 %.3f (재현 %.3f -> %.3f)" % (c[:32], 속, 지금[c]["F@2mm"], f, 지금[c]["재현"], 재), flush=True)
    med = lambda k: float(np.median([v[k] for v in 기록.values()]))
    print("35장 F@2mm 중앙 %.3f -> 겉 %.3f" % (med("F@2mm"), med("F@2mm 겉")))
    json.dump(기록, open(os.path.join(A.OUT, "겉지표.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
