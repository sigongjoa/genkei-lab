"""**exe 로 형태가 나오나** — 벤치가 조각.exe 를 그대로 부르고, 나온 파일만 채점한다.

    python 벤치/exe벤치.py [도형|마네킹|실물 ...]     -> out/exe벤치.json · out/exe벤치.png

  exe 에 주는 것   시트 그림 두 장(front.png · side.png) + 키 150 mm. 케이스 이름 · 정답은 안 준다.
  exe 가 하는 것   `조각.exe 만들기` = API.그림에서 — 카탈로그 후보를 그림과 대 보고 1순위를 저널 한 칸으로 (조각/그림.py)
  벤치가 보는 것   메시.stl → 3D IoU · 저널.json → 파이썬으로 새로 재생해 해시가 exe 와 같은가 · 결과.json(후보 · 동점 · 앱 점수)

  1순위    exe 가 혼자 고른 것
  고르면   동점(그림 두 장이 못 가른 후보들) 안에서 사람이 골랐을 때의 최선 — 정답을 보고 고르니 상한이다
  앱 점수  exe 가 아는 유일한 점수(입력 그림과 실루엣 IoU). 정답 없이 「못 잡았다」를 아는지 본다

예측은 돌리기 전에 적었다 (09-22, 커밋이 먼저다).

돌린 뒤 (09-22, 46장 · 장당 5~6초) — 여덟 다 맞음:
  E1 46장 모두 exe 해시 = 파이썬 재생 해시.
  도형: 가를 수 있는 여섯은 exe 가 혼자 0.968~1.000 · 네모×네모 넷은 동점 4 · 대각은 동점 2 → 고르면 1.000 · 0.997 · 1.000, L자 0.758.
  마네킹 중앙 0.330 · 실물 중앙 0.365 — 기본형 하나로는 사람형을 못 잡는다(원화3d 파이프라인 실물 0.769). 1순위는 거의 원뿔 · 상자.
  구 그대로(0.05)는 이긴다. 앱 점수 ↔ 진짜 IoU 순위상관 0.695 · 사람형 앱 점수 중앙 0.605 — 앱이 「못 잡았다」를 안다.
  입력 한계: cloak_Cloche · wizard_Evil 은 물체가 창보다 넓어 시트 그림이 잘려 있다. statue 는 얇은 뒷판 + 말이라 상자가 부피를 못 맞춘다(0.028).
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
import dsl as D             # noqa: E402

EXE = os.path.join(HERE, "..", "조각", "dist", "조각.exe")
시트 = os.path.join(A.원화3d, "out", "시트")
묶음 = {
    "도형": A.도형,
    "마네킹": ["마네킹_" + n for n in ("차렷", "A포즈", "T포즈", "팔앞뒤", "다리벌림", "검들기", "치마")],
    "실물": [c["name"] for c in json.load(open(os.path.join(A.원화3d, "벤치.json"), encoding="utf-8"))["cases"] if c["name"] != "소체"],
}
혼자 = ["도형_" + n for n in ("구", "원뿔", "치마", "토러스세움", "토러스옆", "두기둥나란히")]
네모 = ["도형_" + n for n in ("상자", "원기둥", "상자45도", "L자")]

예측 = [
    "E1 exe = 스크립트: 모든 장에서 exe 저널을 파이썬으로 새로 재생한 해시가 exe 메시 해시와 같다",
    "E2 도형 — 앞 · 옆으로 가를 수 있는 여섯(구 · 원뿔 · 치마 · 토러스 둘 · 나란히 두 기둥)은 exe 가 혼자 골라 >= 0.95, 동점 없음",
    "E3 도형 — 네모×네모 넷(상자 · 원기둥 · 45도 상자 · L자)은 동점 >= 3 으로 넘기고, 대각 두 기둥은 동점 2",
    "E4 도형 — 동점 안에서 고르면 상자 · 원기둥 · 45도 상자 >= 0.95, L자 < 0.90",
    "E5 마네킹 7 · 실물 28 — 기본형 다섯으로는 못 잡는다: 1순위 중앙 < 0.60 (원화3d 파이프라인 실물 0.769)",
    "E6 그래도 구 그대로는 이긴다: 마네킹 · 실물 각각 1순위 중앙 > 구 그대로 중앙",
    "E7 앱이 스스로 안다: 전체에서 앱 점수와 진짜 IoU3D 의 순위상관 >= 0.5",
    "E8 사람형(마네킹 · 실물)에서 앱 점수 중앙 < 0.80 — 앱도 못 잡은 걸 안다",
]


def exe로(case, 폴더):
    t = time.time()
    r = subprocess.run([EXE, "만들기", os.path.join(시트, case, "front.png"), os.path.join(시트, case, "side.png"), "150", 폴더],
                       timeout=600)
    if r.returncode:
        raise RuntimeError(open(os.path.join(폴더, "오류.txt"), encoding="utf-8").read())
    return time.time() - t


def 한장(case, 재생):
    폴더 = os.path.join(A.OUT, "exe", case)
    초 = exe로(case, 폴더)
    결과 = json.load(open(os.path.join(폴더, "결과.json"), encoding="utf-8"))
    저널 = json.load(open(os.path.join(폴더, "저널.json"), encoding="utf-8"))
    m = trimesh.load(os.path.join(폴더, "메시.stl"), force="mesh")
    답 = A.정답(case)
    s = A.채점(m.vertices, m.faces, 답)
    e = 재생(저널["저널"])                                         # exe 가 아니라 이 파이썬으로 새로 튼다
    동점 = 결과["동점"]
    고르면 = max([A.채점(*D.실행(h["줄"]), 답)["IoU3D"] for h in 결과["후보"][1:동점]] + [s["IoU3D"]])
    구 = trimesh.creation.icosphere(4, 50.0)
    return {"1순위": 결과["후보"][0]["이름"], "줄": 결과["후보"][0]["줄"], "IoU3D": s["IoU3D"], "앞": s["앞"], "옆": s["옆"],
            "고르면": 고르면, "동점": 동점, "동점 후보": [h["이름"] for h in 결과["후보"][:동점]],
            "앱 점수": 결과["후보"][0]["점수"], "구 그대로": A.채점(구.vertices, 구.faces, 답)["IoU3D"],
            "exe 해시": 결과["해시"], "재생 해시": e.해시(), "exe": 결과["exe"], "초": round(초, 1),
            "_그림": (답, m)}


def main(묶음들):
    import app
    재생 = lambda 칸들: app.API()._재생(칸들)
    기록, 칸들 = {}, []
    for g in 묶음들:
        for c in 묶음[g]:
            r = 한장(c, 재생)
            답, m = r.pop("_그림")
            기록[c] = dict(r, 묶음=g)
            print("  %-4s %-36s %-6s IoU3D %.3f · 고르면 %.3f · 동점 %d · 앱 %.3f · 구 %.3f · 해시 %s · %.0f초" % (
                g, c[:36], r["1순위"], r["IoU3D"], r["고르면"], r["동점"], r["앱 점수"], r["구 그대로"],
                "같음" if r["exe 해시"] == r["재생 해시"] else "다름!", r["초"]), flush=True)
            tV, tF, Vw = np.asarray(답["메시"].vertices), 답["메시"].faces, A.창(m.vertices)
            칸들.append(("%s · %s" % (c.replace("도형_", "").replace("마네킹_", "")[:22], r["1순위"]),
                        [(A.실루엣(tV, tF, v), A.실루엣(Vw, m.faces, v)) for v in ("front", "side")],
                        "3D %.3f · 앱 %.3f · 동점 %d" % (r["IoU3D"], r["앱 점수"], r["동점"]), r["IoU3D"] < 0.6))
    json.dump({"기록": 기록}, open(os.path.join(A.OUT, "exe벤치.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    판정(기록)
    p = A.그림(칸들, "exe 벤치 — 조각.exe 가 그림 두 장만 보고 고른 형태 (회색 겹침 · 빨강 못 채움 · 파랑 넘침)",
              "칸마다 앞 | 옆 · 제목 = 케이스 · exe 1순위 · 앱 = exe 가 아는 점수(입력 그림 대비) · 빨간 제목 = 3D < 0.6", "exe벤치.png", 열=5, S=110)
    print("->", p)


def 판정(기록):
    from scipy.stats import spearmanr
    k = lambda 묶: [v for c, v in 기록.items() if v["묶음"] == 묶]
    med = lambda xs: float(np.median(xs)) if xs else float("nan")
    사람 = k("마네킹") + k("실물")
    print("\n예측")
    결과 = []
    def 적기(i, 맞, 값):
        결과.append({"예측": 예측[i], "맞음": bool(맞), "값": 값})
        print("  %s %s\n      %s" % ("○" if 맞 else "✗", 예측[i], 값))
    적기(0, all(v["exe 해시"] == v["재생 해시"] and v["exe"] for v in 기록.values()), "%d장" % len(기록))
    if all(c in 기록 for c in 혼자 + 네모):
        적기(1, all(기록[c]["IoU3D"] >= 0.95 and 기록[c]["동점"] == 1 for c in 혼자),
             " · ".join("%s %.3f/%d" % (c[3:], 기록[c]["IoU3D"], 기록[c]["동점"]) for c in 혼자))
        적기(2, all(기록[c]["동점"] >= 3 for c in 네모) and 기록["도형_두기둥대각"]["동점"] == 2,
             " · ".join("%s %d" % (c[3:], 기록[c]["동점"]) for c in 네모 + ["도형_두기둥대각"]))
        적기(3, all(기록[c]["고르면"] >= 0.95 for c in 네모[:3]) and 기록["도형_L자"]["고르면"] < 0.9,
             " · ".join("%s %.3f" % (c[3:], 기록[c]["고르면"]) for c in 네모))
    if 사람:
        m1, m2 = med([v["IoU3D"] for v in k("마네킹")]), med([v["IoU3D"] for v in k("실물")])
        적기(4, (np.isnan(m1) or m1 < 0.6) and (np.isnan(m2) or m2 < 0.6), "마네킹 중앙 %.3f · 실물 중앙 %.3f" % (m1, m2))
        b1, b2 = med([v["구 그대로"] for v in k("마네킹")]), med([v["구 그대로"] for v in k("실물")])
        적기(5, (np.isnan(m1) or m1 > b1) and (np.isnan(m2) or m2 > b2), "구 그대로 마네킹 %.3f · 실물 %.3f" % (b1, b2))
        ρ = spearmanr([v["앱 점수"] for v in 기록.values()], [v["IoU3D"] for v in 기록.values()])[0]
        적기(6, ρ >= 0.5, "순위상관 %.3f (%d장)" % (ρ, len(기록)))
        a = med([v["앱 점수"] for v in 사람])
        적기(7, a < 0.8, "사람형 앱 점수 중앙 %.3f" % a)
    json.dump({"기록": 기록, "예측": 결과}, open(os.path.join(A.OUT, "exe벤치.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["도형", "마네킹", "실물"])
