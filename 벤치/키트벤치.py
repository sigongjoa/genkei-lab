"""키트 벤치 — `조각.exe 키트` 로 그림 두 장에서 출력 키트까지. 사람형 35 + 도형 11.

    python 벤치/키트벤치.py            -> out/키트벤치.json · out/키트벤치.png · out/exe/<케이스>/{부품/*.stl, 키트.json, 키트.png}
    python 벤치/키트벤치.py 이어서 10   끝난 장은 건너뛰고 다음 10 장만 (메모리가 모자라 백그라운드가 죽을 때 — 09-23)
    python 벤치/키트벤치.py 판정        적힌 기록으로 판정 · 그림만

  1순위  exe 가 고른 형태(메시.stl) 의 3D IoU
  혼합 · 층 타원  exe 결과.json 후보 줄을 파이썬으로 실행해 잰 3D IoU (같은 dsl.py)
  키트   exe 가 쓴 키트.json 의 판정 Q1~Q6 · 부품 부피 합 / 형태 부피

개발은 두 장(마네킹_T포즈 · avatarsample_d)에서만: 층 타원 0.870 · 0.827 → 혼합 0.906 · 0.862. 키트 검사는 코드로 그린 사람.
예측 (09-22, 돌리기 전에 커밋):
  K1 사람형 35장 혼합 중앙 > 층 타원 중앙, 그리고 >= 0.76
  K2 exe 1순위 사람형 중앙 >= 0.76
  K3 도형 가를 수 있는 여섯 그대로 >= 0.95
  K4 키트 Q1 닫힘 · Q4 부품 <= 15 · Q5 다 이어짐 — 46장 모두 통과
  K5 부품 부피 합 / 형태 부피 — 46장 모두 0.98 ~ 1.02 (핀 공차만큼만 다르다)
  K6 Q6 자립 — 받침 없이 서는 장이 사람형의 60% 이상, 받침을 더한 뒤로는 46장 모두 통과
  K7 Q3 얇은 부피 — 사람형의 50% 이상이 통과
  K8 46장 exe 해시 = 파이썬 재생 해시

다보 모양 (09-23, README 「다음」 1) — 둥근 핀은 돌아가고 좌우가 같았다. 사다리꼴(왼) · 뒤집힌 사다리꼴(오)로 바꾸고 Q7 을 더했다.
예측 (다시 돌리기 전에 커밋):
  D1 다보가 있는 장은 모두 Q7 통과 (90° 돌리면 밖으로 >= 5% · 왼 다보는 오른 구멍에 >= 5% 남는다)
  D2 Q1 · Q4 46/46 · Q5 44/46 · Q6 46/46 — 전과 같다 (다보는 판정을 못 바꾼다)
  D3 부피비 0.98 ~ 1.02 · D4 사람형 exe 1순위 중앙 0.744 ± 0.005 (다보는 모양을 거의 안 바꾼다) · D5 해시 46장
  돌린 뒤 (09-23, 46장): D1 ✗ 44/46 — 검들기 · monster 에서 좌우 다보 **크기**가 달라(팔 2.33 대 4.0) 작은 쪽이 큰 구멍에
  들어갔다. 모양은 맞았다. D2 · D3 · D5 ○ · D4 ○ 0.744 그대로.
  고침: 좌우 짝은 다보 크기를 큰 쪽으로 맞춘다 → 모양만으로 갈린다. 다시 돌리기 전 기대: D6 Q7 46/46 · 나머지는 그대로.
  다시 돌린 뒤 (09-23): D6 ○ — 핀이 있는 35장 모두 Q7 통과(90° 돌리면 밖 0.14~0.22 · 왼 다보 대 오른 구멍 0.12~0.15).
  나머지도 그대로: Q1 · Q4 46/46 · Q5 44/46 · Q6 46/46 · 부피비 0.998~1.004 · 사람형 exe 중앙 0.744 · 해시 46장.

돌린 뒤 (09-23, 46장 — 메모리가 모자라 백그라운드가 죽어 10 장씩 이어서) — 여덟 중 다섯:
  ○ K3 도형 여섯 그대로 · ○ K5 부피비 0.998~1.003 · ○ K6 받침 없이 서는 사람형 77%, 받침 뒤 46/46 · ○ K7 Q3 통과 54% · ○ K8 해시 46장
  ✗ K1 혼합 0.754 (혼합 후보가 있는 31장) · 층 타원 0.745 (35장) — 0.76 에 못 미침. 같은 31장 짝으로는 0.754 대 0.754, 혼합이 이긴 장 21.
  ✗ K2 exe 1순위 중앙 0.744 — 마네킹 0.870 → 0.906 는 올랐지만 실물은 앱 점수가 나은 쪽을 늘 고르진 못한다(31장 중 26장만)
  ✗ K4 Q5 다 이어짐 44/46 — robot_ejDr8 · wizard_o87 에 몸통에 핀이 없는 부품이 남는다
  더 본 것: 마네킹 7장은 모두 Q2 (접합면 반지름 < 3 mm, 목 · 손목) · 사람 모양이 아닌 것(모자 · 상자 · 종 · 메카 · 조각상)은
  사람 규칙으로 잘려 부품이 엉뚱하다. 도중에 고친 것: 얇은몫 복셀화(cloak_Chest 메모리) · 기본형은 한 부품 · 빈 부품 뺌.
  사람형 35장은 마지막 수정(기본형 한 부품) 전 exe 로 돌았다 — 사람형 후보는 그 수정의 영향을 안 받는다.
"""
import json
import os
import subprocess
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
import dsl as D             # noqa: E402
import exe벤치 as X          # noqa: E402


def 한장(case, 재생):
    폴더 = os.path.join(A.OUT, "exe", case)
    r = subprocess.run([X.EXE, "키트", os.path.join(X.시트, case, "front.png"), os.path.join(X.시트, case, "side.png"), "150", 폴더], timeout=1800)
    if r.returncode:
        raise RuntimeError(open(os.path.join(폴더, "오류.txt"), encoding="utf-8").read())
    결과 = json.load(open(os.path.join(폴더, "결과.json"), encoding="utf-8"))
    저널 = json.load(open(os.path.join(폴더, "저널.json"), encoding="utf-8"))
    키트 = json.load(open(os.path.join(폴더, "키트.json"), encoding="utf-8"))
    답 = A.정답(case)
    m = trimesh.load(os.path.join(폴더, "메시.stl"), force="mesh")
    out = {"1순위": 결과["후보"][0]["이름"], "exe": A.채점(m.vertices, m.faces, 답)["IoU3D"],
           "해시 같음": 재생(저널["저널"]).해시() == 결과["해시"],
           "판정": {k: v["통과"] for k, v in 키트["판정"].items()}, "받침": 키트["받침 더함"],
           "부품": sorted(키트["부품"]), "핀": len(키트["핀"]),
           "부피비": round(sum(v["부피 mm3"] for k, v in 키트["부품"].items() if k != "받침") / abs(m.volume), 4)}
    for 이름 in ("혼합", "층 타원"):
        h = next((h for h in 결과["후보"] if h["이름"] == 이름), None)
        if h:
            out[이름] = A.채점(*D.실행(h["줄"]), 답)["IoU3D"]
    return out


def main(몇=None):
    import app
    재생 = lambda 칸들: app.API()._재생(칸들)
    p = os.path.join(A.OUT, "키트벤치.json")
    기록 = json.load(open(p, encoding="utf-8"))["기록"] if 몇 and os.path.exists(p) else {}
    for g in ("마네킹", "실물", "도형"):
        for c in X.묶음[g]:
            if c in 기록:
                continue
            if 몇 is not None:
                if 몇 == 0:
                    남은 = sum(1 for gg in ("마네킹", "실물", "도형") for cc in X.묶음[gg] if cc not in 기록)
                    print("  %d 장 남음" % 남은)
                    return
                몇 -= 1
            r = 한장(c, 재생)
            기록[c] = dict(r, 묶음=g)
            print("  %-4s %-36s %-6s 3D %.3f · 혼합 %s · 층 %s · 부품 %d · 핀 %d · 부피비 %.3f · %s%s · 해시 %s" % (
                g, c[:36], r["1순위"], r["exe"], "%.3f" % r["혼합"] if "혼합" in r else "-", "%.3f" % r["층 타원"] if "층 타원" in r else "-",
                len(r["부품"]), r["핀"], r["부피비"], "".join("○" if v else "×" for v in r["판정"].values()), " 받침" if r["받침"] else "",
                "같음" if r["해시 같음"] else "다름!"), flush=True)
            json.dump({"기록": 기록}, open(os.path.join(A.OUT, "키트벤치.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    판정(기록)
    모음(기록)


def 판정(기록):
    사람 = [v for v in 기록.values() if v["묶음"] != "도형"]
    med = lambda k: float(np.median([v[k] for v in 사람 if k in v]))
    q = lambda v, 앞: next(b for k, b in v["판정"].items() if k.startswith(앞))
    옛 = ["도형_" + n for n in ("구", "원뿔", "치마", "토러스세움", "토러스옆", "두기둥나란히")]
    서 = np.mean([not v["받침"] for v in 사람])
    값 = [(med("혼합") > med("층 타원") and med("혼합") >= 0.76, "혼합 %.3f · 층 타원 %.3f" % (med("혼합"), med("층 타원"))),
          (med("exe") >= 0.76, "exe 1순위 중앙 %.3f · 고른 것 %s" % (med("exe"), sorted({v["1순위"] for v in 사람}))),
          (all(기록[c]["exe"] >= 0.95 for c in 옛), " · ".join("%s %.3f" % (c[3:], 기록[c]["exe"]) for c in 옛)),
          (all(q(v, "Q1") and q(v, "Q4") and q(v, "Q5") for v in 기록.values()), "Q1 %d · Q4 %d · Q5 %d / %d" % tuple(
              [sum(q(v, k) for v in 기록.values()) for k in ("Q1", "Q4", "Q5")] + [len(기록)])),
          (all(0.98 <= v["부피비"] <= 1.02 for v in 기록.values()), "부피비 %.3f ~ %.3f" % (min(v["부피비"] for v in 기록.values()), max(v["부피비"] for v in 기록.values()))),
          (서 >= 0.6 and all(q(v, "Q6") for v in 기록.values()), "받침 없이 서는 사람형 %.0f%% · Q6 통과 %d/%d" % (100 * 서, sum(q(v, "Q6") for v in 기록.values()), len(기록))),
          (np.mean([q(v, "Q3") for v in 사람]) >= 0.5, "사람형 Q3 통과 %.0f%%" % (100 * np.mean([q(v, "Q3") for v in 사람]))),
          (all(v["해시 같음"] for v in 기록.values()), "%d장" % len(기록))]
    예측 = [l.strip() for l in __doc__.splitlines() if l.strip().startswith("K")]
    print("\n예측")
    결과 = []
    for 말, (맞, 글) in zip(예측, 값):
        print("  %s %s\n      %s" % ("○" if 맞 else "✗", 말, 글))
        결과.append({"예측": 말, "맞음": bool(맞), "값": 글})
    for g in ("마네킹", "실물"):
        xs = [v for v in 사람 if v["묶음"] == g]
        print("  %s %d장 중앙 — 혼합 %.3f (%d장) · 층 타원 %.3f · exe %.3f" % (g, len(xs), float(np.median([v["혼합"] for v in xs if "혼합" in v])),
              sum("혼합" in v for v in xs), *[float(np.median([v[k] for v in xs])) for k in ("층 타원", "exe")]))
    짝 = [v for v in 사람 if "혼합" in v]
    print("  짝 비교(혼합 후보가 있는 %d장): 혼합 %.3f · 층 타원 %.3f · 혼합이 이긴 장 %d · exe 가 둘 중 나은 쪽을 고른 장 %d" % (
        len(짝), np.median([v["혼합"] for v in 짝]), np.median([v["층 타원"] for v in 짝]), sum(v["혼합"] > v["층 타원"] for v in 짝),
        sum(v["exe"] >= max(v["혼합"], v["층 타원"]) - 1e-9 for v in 짝)))
    json.dump({"기록": 기록, "예측": 결과}, open(os.path.join(A.OUT, "키트벤치.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def 모음(기록, 열=6, 칸=(260, 320)):
    """케이스마다 키트.png 의 분해도 쪽을 한 장에."""
    cs = [c for c, v in 기록.items() if v["묶음"] != "도형"]
    im = Image.new("RGB", (열 * 칸[0], ((len(cs) + 열 - 1) // 열) * (칸[1] + 36)), "white")
    d = ImageDraw.Draw(im)
    import 키트 as K
    for i, c in enumerate(cs):
        k = Image.open(os.path.join(A.OUT, "exe", c, "키트.png")).crop((560, 60, 1080, 700))
        k.thumbnail(칸)
        x, y = (i % 열) * 칸[0], (i // 열) * (칸[1] + 36)
        im.paste(k, (x, y))
        v = 기록[c]
        d.text((x + 6, y + 칸[1] + 2), "%s · %.2f · 부품 %d%s" % (c.replace("마네킹_", "")[:18], v["exe"], len(v["부품"]), " · 받침" if v["받침"] else ""),
               fill="black" if all(v["판정"].values()) else (200, 60, 50), font=K._글꼴(13))
    im.save(os.path.join(A.OUT, "키트벤치.png"))
    print("->", os.path.join(A.OUT, "키트벤치.png"))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "이어서":
        main(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "판정":
        기록 = json.load(open(os.path.join(A.OUT, "키트벤치.json"), encoding="utf-8"))["기록"]
        판정(기록)
        모음(기록)
    else:
        main()
