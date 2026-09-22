"""도형 11 — 대조군 두 줄. 조각 앱이 아무것도 안 했을 때와, 정점만 옮겨서 갈 수 있는 대략의 끝.

    python 벤치/대조군.py        -> out/대조군.json · out/대조군.png

  구 그대로   조각 앱 기본 구(반지름 50 mm, 세분 6) 를 손대지 않고 잰다. 모든 다음 판본이 이겨야 할 바닥.
  수축포장    구의 정점을 정답 표면의 가장 가까운 점으로 옮긴다 — 면 연결(위상)은 그대로.
              붓이 정점만 옮기는 한 대략 여기까지다. ponytail: 가장 가까운 점 투영은 진짜 최적이 아니다 (접힘이 생긴다).
    안       기본 구(반지름 50) 그대로에서 투영. **첫 판(09-22) — 상한이 못 된다**: 안에서 나가면 모서리 띠가 비어
             볼록한 상자도 0.73 (P3 틀림). 그래서 토러스 0.71 도 위상 탓인지 방법 탓인지 못 가른다.
    밖       같은 구를 정답을 감싸게 키워(위상 같음) 투영 — 표준 수축포장. 예측은 그대로 두고 이 판에도 잰다.

예측은 돌리기 전에 적었다 (09-22, 커밋이 먼저다). 맞았나 틀렸나는 돌린 뒤 옆에 적는다.

돌린 뒤 (09-22, 밖 판):
  볼록 다섯 · 치마 0.996~1.000 — 정점만 옮겨도 된다.
  토러스 둘 0.859 — 구멍 자리에 막. **위상 벽**: 정점을 어떻게 옮겨도 구에서 구멍은 안 생긴다.
  두 기둥 0.478 · 0.779 — 사이에 막. **위상 벽**: 한 덩어리에서 두 덩어리는 안 나온다.
  L자 0.858 — 오목한 안쪽 모서리에 막. 이건 위상 벽이 아니라 가장 가까운 점 투영의 벽이다(구와 L자는 위상이 같다).
             앞 · 옆 실루엣은 1.000 — 오목함이 위에서만 보여서 두 판으로는 못 잡는다.
  → 도형 11 로 보면 구멍 뚫기 · 덩어리 나누기(또는 기본형 여럿 조합)가 필요하다. 붓 · Dyntopo 로는 안 된다.
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
    안, _, _ = trimesh.proximity.closest_point(답mm, V)
    R = np.linalg.norm(답mm.vertices, axis=1).max() * 1.05            # 답mm 은 가운데에 있다
    밖, _, _ = trimesh.proximity.closest_point(답mm, V * (R / np.linalg.norm(V, axis=1).max()))
    return [("구 그대로", 조각(V, F)), ("수축포장 안", 조각(안, F)), ("수축포장 밖", 조각(밖, F))]


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
        쌍들.append((A.실루엣(tV, tF, "side"), A.실루엣(Vw, e.F, "side")))      # 마지막 판본(수축포장 밖) 옆
        칸들.append((c, 쌍들, "구 %s · 안 %s · 밖 %s" % tuple(글), float(글[2]) < 0.9))

    print("\n예측")
    결과 = []
    for 말, 판0, 칸, op, 문턱 in 예측:
      for 판 in ([판0] if 판0 in 기록 else [판0 + " 안", 판0 + " 밖"]):
        말판 = 말 if 판 == 판0 else "%s [%s]" % (말, 판[-1])
        값 = [기록[판][k]["IoU3D"] for k in 칸] if 칸 != "중앙" else [float(np.median([v["IoU3D"] for v in 기록[판].values()]))]
        맞 = all((v >= 문턱) if op == ">=" else (v < 문턱) for v in 값)
        결과.append({"예측": 말판, "판본": 판, "값": 값, "문턱": "%s %.2f" % (op, 문턱), "맞음": 맞})
        print("  %s %s — %s %s %.2f" % ("○" if 맞 else "✗", 말판, " · ".join("%.3f" % v for v in 값), op, 문턱))

    os.makedirs(A.OUT, exist_ok=True)
    json.dump({"기록": 기록, "예측": 결과}, open(os.path.join(A.OUT, "대조군.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    p = A.그림(칸들, "도형 11 대조군 — 회색 겹침 · 빨강 정답만(못 채움) · 파랑 조각만(넘침)",
              "칸마다 앞: 구 그대로 | 수축포장 안 | 수축포장 밖 · 그리고 밖의 옆 · 빨간 제목 = 수축포장 밖 < 0.9", "대조군.png", 열=3)
    print("->", p)


if __name__ == "__main__":
    main()
