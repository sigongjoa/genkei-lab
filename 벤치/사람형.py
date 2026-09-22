"""사람형 A/B/C — exe 가 낸 후보 가운데 셋을 따로 잰다 (외부 평가 1회 설계 · 외부평가/README.md).

    python 벤치/사람형.py        -> out/사람형.json · out/사람형.png   (조각/dist/조각.exe 를 먼저 굽는다)

  A  부위 타원뿔대   부위(머리 · 몸통 · 팔 · 다리)마다 단면 둘
  B  부위 로프트     부위마다 단면 다섯 (0 · 25 · 50 · 75 · 100 %)          ← 본 판
  C  층 타원         부위 없이 높이 60 켜, 켜마다 앞 구간 하나에 타원 하나  ← 대조군
  exe              위 셋과 기본형들 가운데 앱 점수로 exe 가 고른 1순위

셋 다 exe 결과.json 에 적힌 DSL 줄을 그대로 실행해 잰다(같은 dsl.py). exe 1순위는 exe 가 쓴 메시.stl 로 잰다.

개발은 두 장(마네킹_T포즈 · avatarsample_d)에서만 했다. 거기서 고친 것 하나: 옆구리가 몸통 중앙 폭보다 넓은 줄의 부스러기가
팔로 잡혀 팔 단면 하나가 엉덩이까지 57 mm 로 늘어났다 → 팔은 가장 큰 연결 덩어리만 남긴다. 그 뒤 두 장: A 0.746 · 0.562 ·
B 0.876 · 0.740 · C 0.870 · 0.827. 예측은 그 다음에, 46장을 돌리기 전에 적었다 (09-22, 커밋이 먼저다).
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
import dsl as D             # noqa: E402
import exe벤치 as X          # noqa: E402

판 = {"A": "부위 타원뿔대", "B": "부위 로프트", "C": "층 타원"}

예측 = [
    "H1 exe = 스크립트: 46장 모두 exe 해시 = 파이썬 재생 해시",
    "H2 (GPT) 사람형 35장 B 중앙 >= 0.60",
    "H3 (GPT) 사람형 35장 B 중앙 - A 중앙 >= +0.15",
    "H4 (GPT) 사람형 35장 C 중앙 - B 중앙 <= 0.05",
    "H5 (우리) 사람형 35장 C 중앙 > B 중앙 — 층 타원이 숫자로는 이긴다",
    "H6 (우리) exe 1순위 사람형 중앙 >= 0.60 (지난번 0.35)",
    "H7 (우리) 도형 — 가를 수 있는 여섯은 새 후보가 들어와도 exe 1순위 >= 0.95",
    "H8 (우리) 앱 점수 ↔ 진짜 IoU 순위상관 >= 0.5 (46장, exe 1순위)",
]


def 한장(case, 재생):
    폴더 = os.path.join(A.OUT, "exe", case)
    초 = X.exe로(case, 폴더)
    결과 = json.load(open(os.path.join(폴더, "결과.json"), encoding="utf-8"))
    저널 = json.load(open(os.path.join(폴더, "저널.json"), encoding="utf-8"))
    답 = A.정답(case)
    m = trimesh.load(os.path.join(폴더, "메시.stl"), force="mesh")
    r = {"exe": A.채점(m.vertices, m.faces, 답)["IoU3D"], "1순위": 결과["후보"][0]["이름"], "앱": 결과["후보"][0]["점수"],
         "동점": 결과["동점"], "해시 같음": 재생(저널["저널"]).해시() == 결과["해시"], "초": round(초, 1)}
    for k, 이름 in 판.items():
        h = next((h for h in 결과["후보"] if h["이름"] == 이름), None)
        if h:
            V, F = D.실행(h["줄"])
            r[k], r[k + " 앱"] = A.채점(V, F, 답)["IoU3D"], h["점수"]
            r["_" + k] = (V, F)
    return r, 답


def main():
    import app
    재생 = lambda 칸들: app.API()._재생(칸들)
    기록, 칸들 = {}, []
    for g in ("도형", "마네킹", "실물"):
        for c in X.묶음[g]:
            r, 답 = 한장(c, 재생)
            tV, tF = np.asarray(답["메시"].vertices), 답["메시"].faces
            if g != "도형":
                쌍 = [(A.실루엣(tV, tF, "front"), A.실루엣(A.창(r["_" + k][0]), r["_" + k][1], "front")) for k in "ABC" if "_" + k in r]
                칸들.append(("%s · exe %s" % (c.replace("마네킹_", "")[:20], r["1순위"]), 쌍,
                            "A %.2f · B %.2f · C %.2f · exe %.2f" % (r.get("A", 0), r.get("B", 0), r.get("C", 0), r["exe"]), r.get("B", 0) < 0.6))
            기록[c] = {k: v for k, v in r.items() if not k.startswith("_")} | {"묶음": g}
            print("  %-4s %-36s A %.3f · B %.3f · C %.3f · exe %.3f (%s, 앱 %.3f, 동점 %d) · 해시 %s · %.0f초" % (
                g, c[:36], r.get("A", float("nan")), r.get("B", float("nan")), r.get("C", float("nan")), r["exe"], r["1순위"], r["앱"],
                r["동점"], "같음" if r["해시 같음"] else "다름!", r["초"]), flush=True)
    판정(기록)
    p = A.그림(칸들, "사람형 A/B/C — 앞에서 본 겹침 (회색 겹침 · 빨강 못 채움 · 파랑 넘침)",
              "칸마다 A 부위 타원뿔대 | B 부위 로프트 5 단면 | C 층 타원 · 빨간 제목 = B < 0.6", "사람형.png", 열=4, S=110)
    print("->", p)


def 판정(기록):
    from scipy.stats import spearmanr
    사람 = [v for v in 기록.values() if v["묶음"] != "도형"]
    med = lambda k: float(np.median([v[k] for v in 사람 if k in v]))
    a, b, c, e = med("A"), med("B"), med("C"), med("exe")
    도형 = ["도형_" + n for n in ("구", "원뿔", "치마", "토러스세움", "토러스옆", "두기둥나란히")]
    ρ = spearmanr([v["앱"] for v in 기록.values()], [v["exe"] for v in 기록.values()])[0]
    값 = [(all(v["해시 같음"] for v in 기록.values()), "%d장" % len(기록)),
          (b >= 0.60, "B 중앙 %.3f" % b),
          (b - a >= 0.15, "A %.3f → B %.3f (%+.3f)" % (a, b, b - a)),
          (c - b <= 0.05, "C %.3f - B %.3f = %+.3f" % (c, b, c - b)),
          (c > b, "C %.3f · B %.3f" % (c, b)),
          (e >= 0.60, "exe 1순위 중앙 %.3f · 고른 것: %s" % (e, ", ".join("%s %d" % (k, n) for k, n in
                                                             sorted({v["1순위"]: sum(1 for w in 사람 if w["1순위"] == v["1순위"]) for v in 사람}.items(), key=lambda x: -x[1])))),
          (all(기록[k]["exe"] >= 0.95 for k in 도형), " · ".join("%s %.3f(%s)" % (k[3:], 기록[k]["exe"], 기록[k]["1순위"]) for k in 도형)),
          (ρ >= 0.5, "순위상관 %.3f" % ρ)]
    print("\n예측")
    결과 = []
    for 말, (맞, 글) in zip(예측, 값):
        print("  %s %s\n      %s" % ("○" if 맞 else "✗", 말, 글))
        결과.append({"예측": 말, "맞음": bool(맞), "값": 글})
    for g in ("마네킹", "실물"):
        xs = [v for v in 사람 if v["묶음"] == g]
        print("  %s %d장 중앙 — A %.3f · B %.3f · C %.3f · exe %.3f" % (g, len(xs), *[float(np.median([v[k] for v in xs])) for k in ("A", "B", "C", "exe")]))
    json.dump({"기록": 기록, "예측": 결과}, open(os.path.join(A.OUT, "사람형.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
