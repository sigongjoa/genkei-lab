"""기본 도형 늘리기 — 새 기본형 여섯(타원체 · 캡슐 · 반구 · 각뿔대 · 둥근상자 · 관)과 그걸 재는 새 도형 일곱.

    python 벤치/도형2.py 시트     정답 일곱을 짓고 시트 그림(앞 · 옆 실루엣)을 굽는다 -> out/시트/도형2_*
    python 벤치/도형2.py          exe 로 일곱 + 도형 11 + 사람형 35 (되돌림 확인) -> out/도형2.json · out/도형2.png

정답은 **trimesh 로 따로** 짓는다(우리 dsl 로 지으면 자기가 자기를 맞히는 셈). 키 150 mm, 조각 좌표로 지어 어댑터로 창에 앉힌다.
시트 그림 = 창 좌표 실루엣(win.silhouette) — 원화3d 시트의 front.png 가 그것과 IoU 1.000 이었다(어댑터 검사).

예측 (09-22, 돌리기 전에 커밋):
  N1 새 일곱 중 여섯(타원체 · 캡슐 · 반구 · 쐐기 · 둥근상자 · 관)은 exe 1순위 3D >= 0.95
  N2 사각뿔은 원뿔과 앞 · 옆이 같다 — 동점 >= 2, 1순위는 목록 앞의 원뿔이라 < 0.95, 동점에서 고르면(각뿔대) >= 0.95
  N3 기존 도형 가를 수 있는 여섯은 그대로 >= 0.95 (새 후보가 가로채지 않는다)
  N4 사람형 35장 exe 1순위 중앙 >= 0.74 (지난번 0.745 — 새 기본형이 층 타원을 가로채지 않는다)
  N5 모든 장 exe 해시 = 파이썬 재생 해시

돌린 뒤 (09-22, 53장) — 다섯 중 셋:
  ○ N1 새 여섯 0.991~1.000, 모두 제 기본형을 골랐다 (쐐기 = 각뿔대).  ○ N4 사람형 중앙 0.745 그대로 (35장 모두 층 타원).  ○ N5 53장 해시 같음.
  ✗ N2 사각뿔은 1순위부터 각뿔대(1.000) — 원뿔과 동점(0.005 안)인데 점수가 조금 높았다.
  ✗ N3 **되돌림 하나**: 치마(속 빈 원뿔)가 이제 각뿔대(0.788)를 골랐다 — 원뿔(0.996)과 동점인데 각뿔대 점수가 조금 높았다.
  → 원뿔 · 사각뿔은 앞 · 옆이 같아 그림 두 장으로는 못 가른다. 지금은 동점 안 1순위를 **그리기 오차(화소)가 정한다** — 동전이다.

동점 규칙 「가」(09-22 사용자) — 동점 안 1순위 = 카탈로그 순서. 다시 돌리기 전 기대(커밋이 먼저):
  G1 치마 1순위 원뿔 >= 0.95 로 돌아온다   G2 사각뿔 1순위 원뿔 < 0.95 · 고르면 >= 0.95   G3 나머지 1순위 · 사람형 0.745 · 해시는 그대로
"""
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
import dsl as D             # noqa: E402
import exe벤치 as X          # noqa: E402

뿌리 = os.path.join(A.OUT, "시트")


def _점hull(P):
    return trimesh.convex.convex_hull(np.asarray(P, float))


def 정답들():
    """이름 -> 조각 좌표 mm 메시 (바닥 z=0 · 키 150). trimesh 로만 짓는다."""
    구 = trimesh.creation.icosphere(5)
    반 = 구.slice_plane([0, 0, 0], [0, 0, 1], cap=True)
    구석 = lambda w, d, h, r: _점hull(np.concatenate([구.vertices * r + [sx * (w / 2 - r), sy * (d / 2 - r), r + sz * (h - 2 * r)]
                                                  for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)]))
    관 = trimesh.creation.annulus(r_min=30, r_max=75, height=60, sections=128)
    관.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    관.apply_translation([0, 0, 75])
    return {
        "타원체": trimesh.Trimesh(구.vertices * [45, 30, 75] + [0, 0, 75], 구.faces),
        "캡슐": trimesh.creation.capsule(height=100, radius=25, count=[64, 64]).apply_translation([0, 0, 75]),
        "반구": trimesh.Trimesh(반.vertices * [85, 85, 150], 반.faces),
        "사각뿔": _점hull([[-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0], [0, 0, 150]]),
        "쐐기": _점hull([[-50, -40, 0], [50, -40, 0], [50, 40, 0], [-50, 40, 0], [-50, 0, 150], [50, 0, 150]]),
        "둥근상자": 구석(100, 70, 150, 15),
        "관": 관,
    }


def 시트():
    for 이름, m in 정답들().items():
        d = os.path.join(뿌리, "도형2_" + 이름)
        os.makedirs(d, exist_ok=True)
        Vw = A.창(m.vertices)
        trimesh.Trimesh(Vw, m.faces, process=False).export(os.path.join(d, "truth.glb"))
        for v in ("front", "side"):
            s = A.실루엣(Vw, m.faces, v)
            Image.fromarray(np.where(s, 0, 255).astype(np.uint8)).save(os.path.join(d, v + ".png"))
        print("  %s — 정점 %d · 닫힘 %s -> %s" % (이름, len(m.vertices), m.is_watertight, d))


def 한장(case, 재생, 시트뿌리):
    폴더 = os.path.join(A.OUT, "exe", case)
    X.exe로(case, 폴더, 시트뿌리)
    결과 = json.load(open(os.path.join(폴더, "결과.json"), encoding="utf-8"))
    저널 = json.load(open(os.path.join(폴더, "저널.json"), encoding="utf-8"))
    답 = A.정답(case, 시트뿌리)
    m = trimesh.load(os.path.join(폴더, "메시.stl"), force="mesh")
    s = A.채점(m.vertices, m.faces, 답)["IoU3D"]
    동점 = 결과["동점"]
    고르면 = max([A.채점(*D.실행(h["줄"]), 답)["IoU3D"] for h in 결과["후보"][1:동점]] + [s])
    return {"exe": s, "고르면": 고르면, "1순위": 결과["후보"][0]["이름"], "동점": 동점,
            "동점 후보": [h["이름"] for h in 결과["후보"][:동점]], "앱": 결과["후보"][0]["점수"],
            "해시 같음": 재생(저널["저널"]).해시() == 결과["해시"]}, 답, m


def main():
    import app
    재생 = lambda 칸들: app.API()._재생(칸들)
    새 = ["도형2_" + n for n in 정답들()]
    기록, 칸들 = {}, []
    for 묶, cases, 시트뿌리 in (("새 도형", 새, 뿌리), ("도형", X.묶음["도형"], None), ("사람형", X.묶음["마네킹"] + X.묶음["실물"], None)):
        for c in cases:
            r, 답, m = 한장(c, 재생, 시트뿌리)
            기록[c] = dict(r, 묶음=묶)
            print("  %-6s %-36s %-8s 3D %.3f · 고르면 %.3f · 동점 %d (%s) · 해시 %s" % (
                묶, c[:36], r["1순위"], r["exe"], r["고르면"], r["동점"], " · ".join(r["동점 후보"]), "같음" if r["해시 같음"] else "다름!"), flush=True)
            if 묶 == "새 도형":
                tV, tF, Vw = np.asarray(답["메시"].vertices), 답["메시"].faces, A.창(m.vertices)
                칸들.append(("%s · %s" % (c[4:], r["1순위"]), [(A.실루엣(tV, tF, v), A.실루엣(Vw, m.faces, v)) for v in ("front", "side", "top")],
                            "3D %.3f · 고르면 %.3f · 동점 %d" % (r["exe"], r["고르면"], r["동점"]), r["exe"] < 0.95))
    k = lambda n: 기록["도형2_" + n]
    여섯 = ["타원체", "캡슐", "반구", "쐐기", "둥근상자", "관"]
    옛 = ["도형_" + n for n in ("구", "원뿔", "치마", "토러스세움", "토러스옆", "두기둥나란히")]
    사람 = float(np.median([v["exe"] for v in 기록.values() if v["묶음"] == "사람형"]))
    판정 = [("N1 새 여섯 1순위 >= 0.95", all(k(n)["exe"] >= 0.95 for n in 여섯), " · ".join("%s %.3f(%s)" % (n, k(n)["exe"], k(n)["1순위"]) for n in 여섯)),
            ("N2 사각뿔 동점 >= 2 · 1순위 < 0.95 · 고르면 >= 0.95", k("사각뿔")["동점"] >= 2 and k("사각뿔")["exe"] < 0.95 and k("사각뿔")["고르면"] >= 0.95,
             "동점 %d (%s) · 1순위 %.3f · 고르면 %.3f" % (k("사각뿔")["동점"], " · ".join(k("사각뿔")["동점 후보"]), k("사각뿔")["exe"], k("사각뿔")["고르면"])),
            ("N3 기존 도형 여섯 >= 0.95", all(기록[c]["exe"] >= 0.95 for c in 옛), " · ".join("%s %.3f(%s)" % (c[3:], 기록[c]["exe"], 기록[c]["1순위"]) for c in 옛)),
            ("N4 사람형 exe 중앙 >= 0.74", 사람 >= 0.74, "중앙 %.3f · 고른 것 %s" % (사람, sorted({v["1순위"] for v in 기록.values() if v["묶음"] == "사람형"}))),
            ("N5 해시", all(v["해시 같음"] for v in 기록.values()), "%d장" % len(기록))]
    print("\n예측")
    for 말, 맞, 글 in 판정:
        print("  %s %s — %s" % ("○" if 맞 else "✗", 말, 글))
    json.dump({"기록": 기록, "예측": [{"예측": a, "맞음": bool(b), "값": t} for a, b, t in 판정]},
              open(os.path.join(A.OUT, "도형2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("->", A.그림(칸들, "새 도형 일곱 — exe 가 그림 두 장만 보고 고른 형태 (회색 겹침 · 빨강 못 채움 · 파랑 넘침)",
                      "칸마다 앞 | 옆 | 위(확인용) · 빨간 제목 = 3D < 0.95", "도형2.png", 열=2, S=120))


if __name__ == "__main__":
    시트() if len(sys.argv) > 1 and sys.argv[1] == "시트" else main()
