"""도형 11 — DSL v0: 구조(기본형 조합)는 주어지고, **수치는 그림에서 잰다**.

    python 벤치/도형DSL.py        -> out/도형DSL.json · out/도형DSL.png

입력은 시트 그림 두 장(front.png · side.png)과 키 150 mm 뿐이다. 정답 메시는 채점에만 쓴다.
구조는 사람이 준다(지금은 케이스 이름 = 카탈로그 항목). 나중에 LLM 이 고른다 — 범주는 LLM, 숫자는 그림.
재는 법은 카탈로그 항목마다 하나 — 폭 · 높이 · 연결 성분 · 안쪽 구멍. 최적화(CMA-ES)는 안 쓴다.

그림 두 장이 못 가르는 것은 **후보를 여럿** 낸다 — 사람이 「이상해요」에서 고르는 몫(기획.md).
  1순위 = AI 가 먼저 내는 것 · 최선 = 후보 중 정답에 가장 가까운 것(사람이 골랐을 때)

예측은 돌리기 전에 적었다 (09-22, 커밋이 먼저다). IoU3D(양쪽 속 채움).

돌린 뒤 (09-22) — 여섯 중 넷 맞음:
  ○ Q1 식별 가능 여덟 0.968 ~ 1.000 (중앙 0.997). 구 그대로 중앙 0.360 · 코드기하(CMA-ES) 0.998 과 비슷한데 최적화 없이 잰 값만으로.
  ○ Q2 토러스 둘 1.000 — 수축포장 0.859 의 위상 벽을 DSL 이 넘었다.
  ✗ Q3 대각 두 기둥, 골라도 0.937. 구조 · 짝은 맞다(앞 · 옆 실루엣 0.996). 폭을 화소 가장자리로 재 1 화소(0.18 mm) 넓고,
       채점 복셀은 1.4 mm · 기둥 폭 30 칸이라 테두리 한 줄씩이 뒤집힌다. 나란히(0.968)도 같은 까닭. 문턱은 안 고친다.
  ○ Q4 45도 상자 1순위 0.522 → 고르면 1.000.  ○ Q5 L자 0.758 (해석값 3/4) — 위 판 없이는 안 된다.
  ✗ Q6 1순위가 구 그대로에 지는 곳 셋: 대각(0.000 · 짝을 거꾸로 골랐다) · 45도 상자(0.522 < 0.556) · 구(0.9978 < 0.9982, 동점).
       → 그림 두 장이 못 가르는 곳에서 1순위는 동전이다. 「이상해요 → 고르기」가 여기서 값을 한다.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
import dsl as D             # noqa: E402

키 = 150.0
RES, U0, U1, Z0, Z1 = A.W.RES, A.W.U0, A.W.U1, A.W.Z0, A.W.Z1
mm = 키 * (U1 - U0) / RES                        # 화소 하나 = 0.176 mm

예측 = [
    ("Q1 식별 가능 여덟은 1순위로 된다", "1순위", ["상자", "원기둥", "원뿔", "구", "두기둥나란히", "치마", "토러스세움", "토러스옆"], ">=", 0.95),
    ("Q2 토러스 — 위상 벽을 넘는다 (수축포장 0.859)", "1순위", ["토러스세움", "토러스옆"], ">=", 0.95),
    ("Q3 두 기둥 대각 — 짝이 둘이라 1순위는 모르지만 고르면 된다", "최선", ["두기둥대각"], ">=", 0.95),
    ("Q4 45도 상자 — 두 장으로는 못 가르지만 고르면 된다", "최선", ["상자45도"], ">=", 0.95),
    ("Q5 L자 — 오목함이 위에서만 보여 두 장으로는 안 된다", "최선", ["L자"], "<", 0.90),
    ("Q6 열한 장 모두 1순위가 구 그대로를 이긴다", "1순위>구", None, None, None),
]


# ── 그림 → 수치 ──────────────────────────────────────────────────────────────

def 마스크(case, view):
    im = np.asarray(Image.open(os.path.join(A.원화3d, "out", "시트", "도형_" + case, view + ".png")).convert("RGBA"))
    return im[..., 3] > 127 if im[..., 3].min() < 255 else im[..., :3].mean(-1) < 250


def 범위(m):
    """마스크 -> (가로 lo, hi, 높이 lo, hi) 창 좌표. 화소 가장자리로 잰다."""
    ys, xs = np.nonzero(m)
    u = lambda c: U0 + c * (U1 - U0) / RES
    z = lambda r: Z1 - r * (Z1 - Z0) / RES
    return u(xs.min()), u(xs.max() + 1), z(ys.max() + 1), z(ys.min())


def 성분들(m):
    lab, n = ndimage.label(m)
    return sorted((범위(lab == i) for i in range(1, n + 1)), key=lambda b: b[0])


def 구멍(m):
    h = ndimage.binary_fill_holes(m) & ~m
    return 범위(h) if h.any() else None


def 줄높이(m):
    """높이마다 반폭 — 원뿔대 맞추기용. -> (z 창, 반폭 창) 배열."""
    rows = np.nonzero(m.any(1))[0]
    w = np.array([m[r].sum() for r in rows]) * (U1 - U0) / RES / 2
    z = Z1 - (rows + 0.5) * (Z1 - Z0) / RES
    return z, w


def f(v):
    return round(float(v) * 키, 1)                 # 창 → mm (키 1 = 150 mm)


def 재기(case):
    """케이스(= 카탈로그 항목) 하나 -> DSL 후보 줄들. 첫째가 1순위. 조각 좌표: y = -(창 y)."""
    앞, 옆 = 마스크(case, "front"), 마스크(case, "side")
    x0, x1, z0, z1 = 범위(앞)
    y0, y1 = 범위(옆)[:2]
    w, d, h, cx, cy = x1 - x0, y1 - y0, z1 - z0, (x0 + x1) / 2, -(y0 + y1) / 2
    놓기 = "x=%s, y=%s, z0=%s" % (f(cx), f(cy), f(z0))
    if case in ("상자", "L자"):
        return ["상자(w=%s, d=%s, h=%s, %s)" % (f(w), f(d), f(h), 놓기)]
    if case == "상자45도":
        s = (w + d) / 2 / np.sqrt(2)                   # 45도 돌린 정사각 기둥이면 앞 폭 = 대각선
        return ["상자(w=%s, d=%s, h=%s, %s)" % (f(w), f(d), f(h), 놓기),
                "상자(w=%s, d=%s, h=%s, %s, rz=45)" % (f(s), f(s), f(h), 놓기)]
    if case == "원기둥":
        return ["원기둥(r=%s, h=%s, %s)" % (f((w + d) / 4), f(h), 놓기)]
    if case == "구":
        return ["구(r=%s, %s)" % (f(h / 2), 놓기)]
    if case in ("원뿔", "치마"):
        z, r = 줄높이(앞)
        k, b = np.polyfit(z, r, 1)                      # 반폭 = k z + b
        return ["원뿔(r아래=%s, r위=%s, h=%s, %s)" % (f(max(k * z0 + b, 0)), f(max(k * z1 + b, 0)), f(h), 놓기)]
    if case in ("토러스세움", "토러스옆"):
        축, 판 = ("y", 앞) if case == "토러스세움" else ("x", 옆)
        Ro, hz = (z1 - z0) / 2, 구멍(판)
        Ri = (hz[3] - hz[2]) / 2
        return ['토러스(R=%s, r=%s, 축="%s", %s)' % (f((Ro + Ri) / 2), f((Ro - Ri) / 2), 축, 놓기)]
    if case in ("두기둥나란히", "두기둥대각"):
        xs, ys = 성분들(앞), 성분들(옆)
        기둥 = lambda bx, by: "상자(w=%s, d=%s, h=%s, x=%s, y=%s, z0=%s)" % (
            f(bx[1] - bx[0]), f(by[1] - by[0]), f(bx[3] - bx[2]), f((bx[0] + bx[1]) / 2), f(-(by[0] + by[1]) / 2), f(bx[2]))
        if len(ys) == 1:                                # 옆에서 겹친다 — 짝이 하나뿐
            return [" + ".join(기둥(b, ys[0]) for b in xs)]
        return [기둥(xs[0], ys[0]) + " + " + 기둥(xs[1], ys[1]),        # 짝 둘 — 그림 두 장으로는 못 가른다
                기둥(xs[0], ys[1]) + " + " + 기둥(xs[1], ys[0])]
    raise KeyError(case)


# ── 돌리기 ───────────────────────────────────────────────────────────────────

def main():
    바닥 = json.load(open(os.path.join(A.OUT, "대조군.json"), encoding="utf-8"))["기록"]["구 그대로"]
    기록, 칸들 = {}, []
    for c in [k.replace("도형_", "") for k in A.도형]:
        답 = A.정답("도형_" + c)
        tV, tF = np.asarray(답["메시"].vertices), 답["메시"].faces
        후보 = []
        for 줄 in 재기(c):
            V, F = D.실행(줄)
            s = A.채점(V, F, 답)
            후보.append({"줄": 줄, "해시": D.해시(V, F), **s, "_V": V, "_F": F})
        best = max(range(len(후보)), key=lambda i: 후보[i]["IoU3D"])
        기록[c] = {"1순위": 후보[0]["IoU3D"], "최선": 후보[best]["IoU3D"], "고른 후보": best, "구 그대로": 바닥[c]["IoU3D"],
                  "후보": [{k: v for k, v in h.items() if not k.startswith("_")} for h in 후보]}
        print("  %-8s 1순위 %.4f · 최선 %.4f (후보 %d개) · 구 그대로 %.4f" % (c, 기록[c]["1순위"], 기록[c]["최선"], len(후보), 바닥[c]["IoU3D"]))
        for i, h in enumerate(후보):
            print("           %s %s" % ("★" if i == best else " ", h["줄"]))
        쌍들 = []
        for h in [후보[0]] + ([후보[best]] if best else []):
            Vw = A.창(h["_V"])
            쌍들 += [(A.실루엣(tV, tF, v), A.실루엣(Vw, h["_F"], v)) for v in ("front", "side", "top")]
        글 = "1순위 %.3f" % 기록[c]["1순위"] + (" · 고르면 %.3f" % 기록[c]["최선"] if best else "")
        칸들.append((c, 쌍들, 글, 기록[c]["최선"] < 0.9))

    print("\n예측")
    결과 = []
    for 말, 판, 칸, op, 문턱 in 예측:
        if 판 == "1순위>구":
            값 = [기록[k]["1순위"] - 기록[k]["구 그대로"] for k in 기록]
            맞 = all(v > 0 for v in 값)
            print("  %s %s — 차이 최소 %+.3f" % ("○" if 맞 else "✗", 말, min(값)))
        else:
            값 = [기록[k][판] for k in 칸]
            맞 = all((v >= 문턱) if op == ">=" else (v < 문턱) for v in 값)
            print("  %s %s — %s %s %.2f" % ("○" if 맞 else "✗", 말, " · ".join("%.3f" % v for v in 값), op, 문턱))
        결과.append({"예측": 말, "값": 값, "맞음": 맞})

    json.dump({"기록": 기록, "예측": 결과}, open(os.path.join(A.OUT, "도형DSL.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    p = A.그림(칸들, "도형 11 · DSL v0 — 그림 두 장에서 잰 수치로 기본형을 놓았다 (회색 겹침 · 빨강 못 채움 · 파랑 넘침)",
              "칸마다 1순위 앞 | 옆 | 위, 후보를 고르면 달라지는 칸은 고른 것 앞 | 옆 | 위 · 입력은 앞 · 옆 두 장뿐, 위는 확인용 · 빨간 제목 = 골라도 < 0.9", "도형DSL.png", 열=2, S=120)
    print("->", p)


if __name__ == "__main__":
    main()
