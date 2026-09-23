"""그림 한 장 + AI 가 만든 옆 그림 -> exe -> 채점 (09-23). 입력 결정: 서비스는 정면 한 장만 받는다.

    python 벤치/생성시야.py <케이스> <생성 시트 png> [꼬리 예: _qwen]     -> out/생성시야/<케이스>/{side_gen.png, 결과.json, 비교.png}

  생성 시트에서 가장 왼쪽 덩어리 = 옆모습(Gemini 는 얼굴이 왼쪽 -> 뒤집어 정답 시트처럼 오른쪽 = 정면).
  같은 exe 로 (정답 옆) · (생성 옆) 두 번 돌려 챔퍼 · F@2mm · 새 각도를 나란히 잰다. 앞 그림은 둘 다 정답 시트 것.
  일관성: 키를 150 mm 로 맞춘 뒤 높이마다 옆 폭(깊이) mm 차이 — 3D 가 아니라 그림끼리.

예측 (09-23, 돌리기 전에 커밋) — avatarsample_d, Gemini 한 장:
  P1 깊이 차 중앙 <= 3 mm (눈으로는 실루엣이 비슷하다)
  P2 생성 옆 F@2mm >= 정답 옆 F@2mm - 0.10 · 챔퍼 중앙 증가 <= 0.5 mm
  P3 위에서 본 그림은 못 쓴다(팔 길이가 뒷모습의 절반) — 이번엔 옆만 쓴다

돌린 뒤 (09-23) — 셋 다 맞음. 단 한 장, 그것도 쉬운 사람형이다:
  깊이 차 중앙 0.79 mm · 90% 2.05 mm · 최대 깊이 26.5 vs 27.3 mm (얼굴 방향만 반대라 뒤집음)
  정답 옆 F@2mm 0.905 · 챔퍼 0.57 · 새 각도 0.937 -> 생성 옆 0.863 · 0.83 · 0.880. 1순위 둘 다 혼합
  위에서 본 그림: 팔이 짧고 머리 위주 — 치수로 못 쓴다. 다음은 안 되는 케이스(상자 · 속 빈 모자)와 여러 장
  cloak_Chest_Closed (Pro 수요 초과 -> Flash 가 그림): 옆 대신 긴 면을 다시 그렸다(둥근 뚜껑 · 고리 없음, 라벨도 FRONT/BACK/TOP).
    최대 깊이 151 -> 275 mm · 깊이 차 중앙 117 mm. F@2mm 0.234 -> 0.160 · 챔퍼 3.27 -> 9.80 mm. 사람이 아닌 물체는 브라우저 생성으로 안 된다

예측 (09-23, 로컬 Qwen-Image-Edit-2511 + Multiple-Angles LoRA, 재기 전에 커밋):
  Q1 avatarsample_d: 깊이 차 중앙 <= 3 mm · F@2mm >= 0.805 (Gemini 와 같은 문턱)
  Q2 cloak_Chest_Closed: 깊이 차 중앙 <= 20 mm (Gemini 117 mm 보다 낫다 — 시야를 각도로 지정하니까)

돌린 뒤 (09-23, 8 GB 3050 · Q4_K_S · 20 걸음 · 걸음당 32초 = 장당 약 11분, --disable-pinned-memory) — Q1 맞음 · Q2 틀림:
  사람형: 깊이 차 중앙 1.11 mm · F@2mm 0.862 · 챔퍼 0.76 · 새 각도 0.893 (Gemini 0.863 · 0.83 · 0.880 과 같은 급)
  상자: 옆이 아니라 3/4 원근, 화면 밖으로 넘침. 깊이 차 27 mm · F 0.150 · 챔퍼 6.36 (정답 옆 0.234 · 3.27)
  결론: 생성 시야는 사람형에서만 쓸 만하다(F -0.04). 사람이 아닌 물체는 두 생성기 다 옆을 못 그린다 — 크레딧보다 앞서 풀 것
"""
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
import 지표 as J             # noqa: E402
import 그림 as G             # noqa: E402
from exe벤치 import exe로, 시트   # noqa: E402


def 옆뽑기(시트png, 뒤집기=True):
    a = np.asarray(Image.open(시트png).convert("RGB")).astype(int)
    m = ndimage.binary_fill_holes(ndimage.binary_closing(a.mean(-1) < 235, iterations=2))
    lab, n = ndimage.label(m)
    큰 = [i + 1 for i, s in enumerate(ndimage.find_objects(lab)) if (lab[s] == i + 1).sum() > 5000]
    i = min(큰, key=lambda k: np.nonzero(lab == k)[1].min())      # 가장 왼쪽
    옆 = lab == i
    return 옆[:, ::-1] if 뒤집기 else 옆


def 깊이줄(m, 층=150):
    """-> 높이 150 칸마다 폭(mm, 키 150). 빈 줄은 0."""
    ys, xs = np.nonzero(m)
    s = 150.0 / (ys.max() + 1 - ys.min())
    out = []
    for k in range(층):
        r = int(ys.max() - (k + 0.5) * (ys.max() + 1 - ys.min()) / 층)
        c = np.nonzero(m[r])[0]
        out.append((c.max() + 1 - c.min()) * s if len(c) else 0.0)
    return np.array(out)


def 재기(case, 폴더):
    m = trimesh.load(os.path.join(폴더, "메시.stl"), force="mesh")
    답 = A.정답(case)["메시"]
    r, _ = J.재기(J.mm메시(A.창(m.vertices), m.faces), J.mm메시(답.vertices, 답.faces))
    return r


def main(case, 시트png, 꼬리=""):
    D = os.path.join(A.OUT, "생성시야", case)
    os.makedirs(D, exist_ok=True)
    옆 = 옆뽑기(시트png)
    rgba = np.zeros(옆.shape + (4,), np.uint8)
    rgba[옆] = (0, 0, 0, 255)
    Image.fromarray(rgba).save(os.path.join(D, "side_gen%s.png" % 꼬리))
    Image.open(os.path.join(시트, case, "front.png")).save(os.path.join(D, "front.png"))
    Image.open(os.path.join(D, "side_gen%s.png" % 꼬리)).save(os.path.join(D, "side.png"))

    진 = G.마스크(os.path.join(시트, case, "side.png"))
    d진, d생 = 깊이줄(진), 깊이줄(옆)
    일관 = {"깊이 차 중앙 mm": round(float(np.median(abs(d진 - d생))), 2),
            "깊이 차 90% mm": round(float(np.percentile(abs(d진 - d생), 90)), 2),
            "최대 깊이 정답 · 생성 mm": [round(float(d진.max()), 1), round(float(d생.max()), 1)]}

    결과 = {"일관성": 일관}
    for 이름, 뿌리 in (("정답 옆", 시트), ("생성 옆", os.path.join(A.OUT, "생성시야"))):
        폴더 = os.path.join(D, "exe_" + 이름.replace(" ", "") + 꼬리)
        exe로(case, 폴더, 뿌리)
        결과[이름] = 재기(case, 폴더)
        결과[이름]["1순위"] = json.load(open(os.path.join(폴더, "결과.json"), encoding="utf-8"))["후보"][0]["이름"]
    json.dump(결과, open(os.path.join(D, "결과%s.json" % 꼬리), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(결과, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
