"""틀 + 칸 라벨 -> 형태 -> F@2mm (09-23). 라벨 정답률은 대리 자 — 진짜 자는 형태다.

    python 벤치/틀빌더.py <LLM 라벨.json>     -> out/틀라벨/빌더.json

  입력은 **정면 한 장**. 켜(60)마다 정면 구간 = 폭, 깊이와 단면 꼴은 가장 가까운 띠의 겹치는 칸 라벨에서:
      깊이  얇음 0.2 · 보통 0.6 · 깊음 1.0 × 폭 (판은 0.15 × 폭)
      꼴    타원 p=2 · 네모 p=8 (층쌓기 초타원) · 판은 타원 · 껍질은 아직 속 찬 타원(속 빔 빌더 없음)
  A 전부 타원 · 보통   B LLM 라벨   C 정답 라벨(등급만 — 라벨 방식의 천장)
  참고: 두 장(정답 옆 그림) exe 결과 = out/지표.json

예측 (09-23, 돌리기 전에 커밋) — GPT 가 답한 5장:
  B1 F@2mm 중앙: B > A
  B2 C > A (정답 라벨이면 라벨 방식이 돕는다 — 아니면 빌더가 문제)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
import 지표 as J             # noqa: E402
import 그림 as G             # noqa: E402
import dsl as D              # noqa: E402

시트 = os.path.join(A.원화3d, "out", "시트")
OUT = os.path.join(A.OUT, "틀라벨")
비 = {"얇음": 0.2, "보통": 0.6, "깊음": 1.0}


def _라벨(칸들, z, a, b, 폭):
    """켜 높이 z(키 비율) · 정면 화소 구간 [a, b) -> (꼴, 깊이 등급). 가장 가까운 띠에서 x 가 겹치는 칸."""
    if not 칸들:
        return "타원", "보통"
    띠 = min({v["z"] for v in 칸들.values()}, key=lambda t: abs(t - z))
    px = lambda x: (x + 0.6) / 1.2 * 폭
    겹 = [(min(b, px(v["x"][1])) - max(a, px(v["x"][0])), v) for v in 칸들.values() if v["z"] == 띠]
    겹 = [t for t in 겹 if t[0] > 0]
    if not 겹:
        return "타원", "보통"
    v = max(겹, key=lambda t: t[0])[1]
    return v.get("단면", "타원"), v.get("깊이", "보통")


def 줄(case, 칸들, 층수=60, 키=150.0):
    앞 = G.마스크(os.path.join(시트, case, "front.png"))
    fa = G.틀(앞, 키)
    rows = np.nonzero(앞.any(1))[0]
    zmax_px = rows.max() + 1 - rows.min()
    경계 = np.linspace(rows.min(), rows.max() + 1, 층수 + 1)
    켜들, 꼴표 = [], {}
    for r0, r1 in zip(경계[:-1], 경계[1:]):
        r = int((r0 + r1) / 2)
        z0, z1 = (fa.bot - r1) * fa.s, (fa.bot - r0) * fa.s
        구간 = G._줄조각(앞[r])
        if not 구간:
            continue
        # 창 z(키 1) 로: 시트는 z ∈ [-0.1, 1.1] -> 1024 행, 바닥 = 물체 바닥
        zr = (fa.bot - r) / zmax_px
        깊, 꼴 = [], []
        for a, b in 구간:
            f, g = _라벨(칸들, zr, a, b, 앞.shape[1])
            k = 0.15 if f == "판" else 비.get(g, 0.6)
            깊.append((b - a) * fa.s * k)
            꼴.append(8.0 if f == "네모" else 2.0)
        켜들.append((z0, z1, 깊, 꼴, 구간))
    켜들 = 켜들[::-1]
    # 그림.켜쌓기줄 과 같게 잇되, 깊이 · 꼴은 구간마다
    토막 = []
    for 줄기 in G.켜줄기([(z0, z1, 0.0, 0.0, 구) for z0, z1, _, _, 구 in 켜들]):
        z, x, y, w, d, p = [], [], [], [], [], []
        for n, (i, (a, b)) in enumerate(줄기):
            z0, z1, 깊, 꼴, 구 = 켜들[i]
            j = 구.index((a, b))
            높이들 = ([z0] if n == 0 else []) + [(z0 + z1) / 2] + ([z1 + 0.01] if n == len(줄기) - 1 else [])
            for h in 높이들:
                z.append(round(float(h), 2)); x.append(round(float(((a + b) / 2 - fa.c) * fa.s), 1)); y.append(0.0)
                w.append(round(float((b - a) * fa.s), 1)); d.append(round(float(깊[j]), 1)); p.append(꼴[j])
        토막.append("층쌓기(z=%s, x=%s, y=%s, w=%s, d=%s, p=%s)" % (z, x, y, w, d, p))
    return " + ".join(토막)


def 재기(case, s):
    V, F = D.실행(s)
    답 = A.정답(case)["메시"]
    r, _ = J.재기(J.mm메시(A.창(V), F), J.mm메시(답.vertices, 답.faces))
    return {k: r[k] for k in ("F@2mm", "챔퍼 중앙", "새 각도 중앙")}


def main(경로):
    정 = json.load(open(os.path.join(OUT, "정답.json"), encoding="utf-8"))
    llm = json.load(open(경로, encoding="utf-8"))
    두장 = json.load(open(os.path.join(A.OUT, "지표.json"), encoding="utf-8"))["기록"]
    기록 = {}
    for n in sorted(llm, key=int):
        c = 정[n]["케이스"]
        r = {"A 타원": 재기(c, 줄(c, {})), "B LLM": 재기(c, 줄(c, {k: dict(v, **llm[n].get(k, {})) for k, v in 정[n]["칸"].items() if k in llm[n]})),
             "C 정답": 재기(c, 줄(c, 정[n]["칸"]))}
        if c in 두장:
            r["두 장"] = {k: 두장[c][k] for k in ("F@2mm", "챔퍼 중앙", "새 각도 중앙")}
        기록[c] = r
        print("%-32s " % c[:32] + " · ".join("%s F %.3f 챔퍼 %.2f" % (k, v["F@2mm"], v["챔퍼 중앙"]) for k, v in r.items()), flush=True)
    med = lambda k: float(np.median([v[k]["F@2mm"] for v in 기록.values()]))
    print("F@2mm 중앙  A %.3f · B %.3f · C %.3f" % (med("A 타원"), med("B LLM"), med("C 정답")))
    print("  %s B1 B > A\n  %s B2 C > A" % ("○" if med("B LLM") > med("A 타원") else "✗", "○" if med("C 정답") > med("A 타원") else "✗"))
    json.dump(기록, open(os.path.join(OUT, "빌더.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
