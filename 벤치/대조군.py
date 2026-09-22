"""도형 11 — 대조군 두 줄. 조각 앱이 아무것도 안 했을 때와, 정점만 옮겨서 갈 수 있는 대략의 끝.

    python 벤치/대조군.py        -> out/대조군.json · out/대조군.png

  구 그대로   조각 앱 기본 구(반지름 50 mm, 세분 6) 를 손대지 않고 잰다. 모든 다음 판본이 이겨야 할 바닥.
  수축포장    구의 정점을 정답 표면의 가장 가까운 점으로 옮긴다 — 면 연결(위상)은 그대로.
              붓이 정점만 옮기는 한 대략 여기까지다. ponytail: 가장 가까운 점 투영은 진짜 최적이 아니다 (접힘이 생긴다).

예측은 돌리기 전에 적었다 (09-22, 커밋이 먼저다). 맞았나 틀렸나는 돌린 뒤 옆에 적는다.
"""
import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
from engine import 조각, 기본메시   # noqa: E402

도형 = [c.replace("도형_", "") for c in A.도형]

# 예측 — (이름, 판본, 칸들, 비교, 문턱). IoU3D(양쪽 속 채움) 기준.
예측 = [
    ("P1 구 그대로가 구에는 거의 맞는다", "구 그대로", ["구"], ">=", 0.98),
    ("P2 구 그대로의 11장 중앙값은 낮다", "구 그대로", "중앙", "<", 0.60),
    ("P3 볼록형은 수축포장으로 거의 된다", "수축포장", ["상자", "원기둥", "원뿔", "구", "상자45도"], ">=", 0.95),
    ("P4 토러스는 구멍을 못 뚫는다 (막이 남는다)", "수축포장", ["토러스세움", "토러스옆"], "<", 0.90),
    ("P5 떨어진 두 기둥 사이에 막이 남는다", "수축포장", ["두기둥대각", "두기둥나란히"], "<", 0.90),
    ("P6 L자 오목한 곳에 막이 남는다", "수축포장", ["L자"], "<", 0.95),
    ("P7 치마는 속을 채우고 재니 겉만 맞으면 된다", "수축포장", ["치마"], ">=", 0.90),
]


def 판본들(case):
    """정답 하나에 대해 (이름, 조각 엔진) 두 개."""
    t = trimesh.load(os.path.join(A.원화3d, "out", "시트", "도형_" + case, "truth.glb"), force="mesh")
    답mm = trimesh.Trimesh(A.조각좌표(t.vertices), t.faces, process=False)
    V, F = 기본메시()
    끝, _, _ = trimesh.proximity.closest_point(답mm, V)
    return [("구 그대로", 조각(V, F)), ("수축포장", 조각(끝, F))]


def main():
    기록, 칸들 = {}, []
    for c in 도형:
        답 = A.정답("도형_" + c)
        tV, tF = np.asarray(답["메시"].vertices), 답["메시"].faces
        쌍들, 글 = [], []
        for 이름, e in 판본들(c):
            s = A.채점(e.V, e.F, 답)
            기록.setdefault(이름, {})[c] = s
            Vw = A.창(e.V)
            쌍들.append((A.실루엣(tV, tF, "front"), A.실루엣(Vw, e.F, "front")))
            글.append("%.3f" % s["IoU3D"])
            print("  %-8s %-6s IoU3D %.4f (점수판 %.4f) · 앞 %.4f · 옆 %.4f" % (c, 이름, s["IoU3D"], s["IoU3D_점수판"], s["앞"], s["옆"]))
        쌍들.append((A.실루엣(tV, tF, "side"), A.실루엣(Vw, e.F, "side")))      # 마지막 판본(수축포장) 옆
        칸들.append((c, 쌍들, "구 그대로 %s · 수축포장 %s" % tuple(글), float(글[1]) < 0.9))

    print("\n예측")
    결과 = []
    for 말, 판, 칸, op, 문턱 in 예측:
        값 = [기록[판][k]["IoU3D"] for k in 칸] if 칸 != "중앙" else [float(np.median([v["IoU3D"] for v in 기록[판].values()]))]
        맞 = all((v >= 문턱) if op == ">=" else (v < 문턱) for v in 값)
        결과.append({"예측": 말, "판본": 판, "값": 값, "문턱": "%s %.2f" % (op, 문턱), "맞음": 맞})
        print("  %s %s — %s %s %.2f" % ("○" if 맞 else "✗", 말, " · ".join("%.3f" % v for v in 값), op, 문턱))

    os.makedirs(A.OUT, exist_ok=True)
    json.dump({"기록": 기록, "예측": 결과}, open(os.path.join(A.OUT, "대조군.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    p = A.그림(칸들, "도형 11 대조군 — 회색 겹침 · 빨강 정답만(못 채움) · 파랑 조각만(넘침)",
              "칸마다: 구 그대로 (앞) | 수축포장 (앞) | 수축포장 (옆) · 빨간 제목 = 수축포장 < 0.9", "대조군.png", 열=3)
    print("->", p)


if __name__ == "__main__":
    main()
