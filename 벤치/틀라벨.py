"""틀은 앱이, 디테일은 LLM 이 (09-23 사용자: 「frame 을 잡아주면 LLM 이 디테일 채우고 디테일 틀리면 CAD 로 고치고」).

    python 벤치/틀라벨.py 틀          케이스마다 정면에 칸(높이 띠 × 구간)을 그린 그림 + 정답 라벨 -> out/틀라벨/
    python 벤치/틀라벨.py 채점 <답.json>   LLM 라벨 {케이스번호: {칸: {단면, 깊이}}} 을 정답과 대 본다

  틀   높이 6 띠(키의 8 · 25 · 42 · 58 · 75 · 92 %). 띠마다 정면 실루엣 한 줄의 구간(연속 화소) = 칸 「3b」.
       정면 한 줄 = 그 높이 수평 단면을 x 로 투영한 것 — 그래서 칸을 정답 메시 단면에서 바로 뽑고 정면 그림에 그린다.
  정답 라벨 (칸의 x 범위에 걸친 단면 다각형들로):
       단면  껍질(안쪽 점의 반 이상에서 위나 아래로 쏜 광선이 안 막힘) > 판(짧은 변 / 긴 변 < 0.25)
             > 네모(돌기 깎은 단면 / 감싸는 축 사각형 >= 0.88) > 타원. 단면 = 닫힌 고리를 칠한 400² 마스크
       깊이  앞뒤(y) / 폭(x): < 0.35 얇음 · < 0.8 보통 · 그 이상 깊음
  LLM 은 그림(정면 한 장 + 칸)만 본다. 숫자는 주지 않고 받지도 않는다 — 범주만.

예측 (09-23, LLM 에 보내기 전에 커밋) — 안 되던 17장 (F@2mm < 0.70), 121칸: 타원 88 · 네모 19 · 판 11 · 껍질 3.
  다 「타원」 찍기 = 0.727 이라 정답률만으로는 모자라다.
  L1 단면 정답률 >= 0.80 (찍기 0.727 보다 위)
  L2 네 범주 평균 재현율 >= 0.55 (찍기 0.25)
  L3 껍질 3칸 중 2칸 이상 맞힘
"""
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402

시트 = os.path.join(A.원화3d, "out", "시트")
OUT = os.path.join(A.OUT, "틀라벨")
띠들 = [0.08, 0.25, 0.42, 0.58, 0.75, 0.92]
케이스 = ["statue_Town_Center_cuQuFJEIfH", "cloak_Chest_Closed_AngpV0HxD8", "mech_Mech_4UvIHxnoSR", "cloak_Chest_O72u4Drp8k",
        "wizard_Wizard_o87Upt5uHX", "cape_Hat_Stylised_lNN3PlrjSa", "cloak_Cloche_Qz9XMYysBQ", "monster_Mimic_B8HrWzkuNp",
        "mech_Drill_8uBbH7Dvmb", "robot_Robot_Enemy_Legs_Gun_lFZfDh2hzP", "wizard_Evil_Wizard_bdxawstqtq", "robot_Robot_ejDr8lRglP",
        "마네킹_검들기", "wizard_WIzard_Gnome_dEuyzEgrF4", "witch_Witch_vWI9PHfjcy", "마네킹_팔앞뒤", "witch_Chibi_Witch_dKexfKQOOl"]


해상 = 400                                                       # 단면 마스크 한 변 (창 x · y 가 [-1, 1])


def _px(v):
    return (np.asarray(v) + 1) / 2 * 해상


def 단면(m, z):
    """-> 닫힌 고리를 칠한 마스크 (y 행 · x 열). 메시가 안 닫혀도 닫힌 고리는 칠한다(= 층마다 구멍 채움)."""
    s = m.section(plane_normal=[0, 0, 1], plane_origin=[0, 0, z])
    M = Image.new("1", (해상, 해상), 0)
    if s is None:
        return np.zeros((해상, 해상), bool)
    d = ImageDraw.Draw(M)
    for q in s.discrete:
        t = [tuple(v) for v in _px(q[:, :2])]
        d.line(t, fill=1, width=2)                                   # 판자처럼 안 닫힌 조각도 선은 긋는다
        if len(q) >= 3 and np.linalg.norm(q[0] - q[-1]) < 1e-3:
            d.polygon(t, fill=1)
    M = ndimage.binary_closing(np.asarray(M), np.ones((3, 3), bool), iterations=3)
    return ndimage.binary_fill_holes(M)


def 라벨(m, z, M):
    """M = 칸 하나의 단면 마스크."""
    ys, xs = np.nonzero(M)
    W, D = xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()
    r0 = max(1, int(min(W, D) / 12))                                   # 장식 돌기를 깎고 꼴을 잰다
    O = ndimage.binary_opening(M, np.ones((2 * r0 + 1, 2 * r0 + 1), bool))
    oy, ox = np.nonzero(O) if O.any() else (ys, xs)
    찬 = len(ox) / ((ox.max() + 1 - ox.min()) * (oy.max() + 1 - oy.min()))
    # 속 빔: 안쪽 점에서 위 · 아래로 쏜 광선이 메시에 안 막히고 나간다 (두께 없는 껍데기도 잡는다)
    k = np.random.default_rng(0).choice(len(xs), min(60, len(xs)), replace=False)
    P = np.stack([xs[k] / 해상 * 2 - 1, ys[k] / 해상 * 2 - 1, np.full(len(k), z)], 1)
    나감 = ~m.ray.intersects_any(P, np.tile([0, 0, -1.0], (len(P), 1))) | ~m.ray.intersects_any(P, np.tile([0, 0, 1.0], (len(P), 1)))
    if 나감.mean() >= 0.5:
        꼴 = "껍질"
    elif min(W, D) < 0.25 * max(W, D):
        꼴 = "판"
    elif 찬 >= 0.88:
        꼴 = "네모"
    else:
        꼴 = "타원"
    r = D / max(W, 1)
    return {"단면": 꼴, "깊이": "얇음" if r < 0.35 else "보통" if r < 0.8 else "깊음", "D/W": round(float(r), 2),
            "찬": round(float(찬), 2), "나감": round(float(나감.mean()), 2)}


def 틀(case):
    m = A.정답(case)["메시"]
    zmax = m.bounds[1, 2]
    칸 = {}
    폭 = m.bounds[1, 0] - m.bounds[0, 0]
    for i, f in enumerate(띠들, 1):
        M = 단면(m, f * zmax)
        열 = np.nonzero(M.any(0))[0]
        if not len(열):
            continue
        끊김 = np.nonzero(np.diff(열) > 1)[0]
        for j, (a, b) in enumerate(zip(np.r_[열[0], 열[끊김 + 1]], np.r_[열[끊김], 열[-1]])):
            x0, x1 = a / 해상 * 2 - 1, (b + 1) / 해상 * 2 - 1
            if (b + 1 - a) / 해상 * 2 < 0.03 * 폭 or min(x1, 0.6) - max(x0, -0.6) < 0.5 * (x1 - x0):
                continue                                             # 가는 것 · 그림 밖으로 반 넘게 잘린 것

            N = np.zeros_like(M)
            N[:, a:b + 1] = M[:, a:b + 1]
            칸["%d%s" % (i, "abcdefgh"[len([k for k in 칸 if k[0] == str(i)])])] = dict(
                라벨(m, f * zmax, N), z=f, x=[x0, x1])
    return m, 칸


def 그리기(case, n, m, 칸):
    im = Image.open(os.path.join(시트, case, "front.png")).convert("RGBA")
    흰 = Image.new("RGBA", im.size, "white")
    im = Image.alpha_composite(흰, im).convert("RGB")
    # 시트 = 원화3d mask2d.topix: 창 u ∈ [-0.6, 0.6] · z ∈ [-0.1, 1.1] -> 1024². 넓은 물체는 잘려 있다
    px = lambda x: (np.clip(x, -0.6, 0.6) + 0.6) / 1.2 * im.width
    zmax = m.bounds[1, 2]
    d = ImageDraw.Draw(im)
    import 키트 as K
    글 = K._글꼴(26)
    for k, v in 칸.items():
        y = (1.1 - v["z"] * zmax) / 1.2 * im.height
        a, b = px(v["x"][0]), px(v["x"][1])
        d.line([(0, y), (im.width, y)], fill=(200, 200, 200), width=1)
        d.line([(a, y), (b, y)], fill=(230, 30, 30), width=5)
        d.text(((a + b) / 2 - 14, y - 32), k, fill=(230, 30, 30), font=글, stroke_width=3, stroke_fill="white")
    d.text((12, 8), "#%d" % n, fill="black", font=K._글꼴(40))
    return im


def 만들기():
    os.makedirs(OUT, exist_ok=True)
    답 = {}
    for n, c in enumerate(케이스, 1):
        m, 칸 = 틀(c)
        그리기(c, n, m, 칸).save(os.path.join(OUT, "%02d.png" % n))
        답[str(n)] = {"케이스": c, "칸": 칸}
        print(n, c, " ".join("%s:%s/%s" % (k, v["단면"], v["깊이"]) for k, v in 칸.items()), flush=True)
    json.dump(답, open(os.path.join(OUT, "정답.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def 채점(경로):
    정 = json.load(open(os.path.join(OUT, "정답.json"), encoding="utf-8"))
    llm = json.load(open(경로, encoding="utf-8"))
    쌍 = [(v["단면"], llm.get(n, {}).get(k, {}).get("단면"), v["깊이"], llm.get(n, {}).get(k, {}).get("깊이"))
         for n, c in 정.items() for k, v in c["칸"].items()]
    from collections import Counter
    다수 = Counter(a for a, *_ in 쌍).most_common(1)[0]
    맞 = np.mean([a == b for a, b, _, _ in 쌍])
    껍 = [b == "껍질" for a, b, _, _ in 쌍 if a == "껍질"]
    깊 = np.mean([c == d for _, _, c, d in 쌍])
    print("칸 %d · 단면 정답률 %.3f · 다수(%s) 찍기 %.3f · 껍질 재현 %s · 깊이 정답률 %.3f" % (
        len(쌍), 맞, 다수[0], 다수[1] / len(쌍), "%.3f (%d칸)" % (np.mean(껍), len(껍)) if 껍 else "정답 껍질 없음", 깊))
    print("혼동 (정답 -> LLM):", dict(Counter((a, b) for a, b, _, _ in 쌍 if a != b)))
    예측 = [l.strip() for l in __doc__.splitlines() if l.strip()[:2] in ("L1", "L2", "L3") and "찍기" in l or l.strip()[:2] == "L3"]
    재현 = [np.mean([b == a for a, b, _, _ in 쌍 if a == 꼴]) for 꼴 in ("타원", "네모", "판", "껍질")]
    print("범주별 재현 타원 %.2f · 네모 %.2f · 판 %.2f · 껍질 %.2f" % tuple(재현))
    값 = [맞 >= 0.8, np.mean(재현) >= 0.55, sum(껍) >= 2]
    for 말, v in zip(예측, 값):
        print("  %s %s" % ("○" if v else "✗", 말))


if __name__ == "__main__":
    만들기() if sys.argv[1] == "틀" else 채점(sys.argv[2])
