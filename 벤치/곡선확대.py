"""이전(계단) 대 곡선 — 면 그늘로 크게 (09-24). 점 구름 갤러리로는 계단이 안 보인다.

    python 벤치/곡선확대.py [이름]  -> out/<이름>확대_*.png (케이스마다 정답 | 이전 exe | out/<이름>, 3/4 · 같은 배율)
"""
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "조각"))
import 어댑터 as A           # noqa: E402
import 키트 as K             # noqa: E402
from 넓은시트 import 잘린    # noqa: E402

새이름 = sys.argv[1] if len(sys.argv) > 1 else "곡선"

케이스 = ["robot_Robot_ejDr8lRglP", "avatarsample_f", "마네킹_다리벌림", "witch_Witch_vWI9PHfjcy", "knight_Warrior_Z6ZUtm6kc1", "wizard_WIzard_Gnome_dEuyzEgrF4",
        "avatarsample_e", "마네킹_치마", "cloak_Cloche_Qz9XMYysBQ"]
뒤집기 = np.array([1.0, -1.0, 1.0])                                     # 창 정면 +y -> 그리기 카메라(-y) 쪽


def 앞보게(V, F):
    return np.asarray(V) * 150 * 뒤집기, np.asarray(F)[:, ::-1]


def main():
    크기 = (360, 440)
    for p in range(0, len(케이스), 3):
        묶 = 케이스[p:p + 3]
        im = Image.new("RGB", (크기[0] * 3 + 20, 40 + len(묶) * (크기[1] + 30)), "white")
        d = ImageDraw.Draw(im)
        d.text((10, 8), "정답 | 이전 exe (계단 켜) | %s — 같은 배율 · 3/4" % 새이름, fill="black", font=K._글꼴(16))
        for k, c in enumerate(묶):
            답 = A.정답(c)["메시"]
            전 = trimesh.load(os.path.join(A.OUT, "exe넓게" if c in 잘린 else "exe", c, "메시.stl"), force="mesh")
            곡 = trimesh.load(os.path.join(A.OUT, 새이름, c, "메시.stl"), force="mesh")
            셋 = [앞보게(답.vertices, 답.faces), 앞보게(A.창(전.vertices), 전.faces), 앞보게(A.창(곡.vertices), 곡.faces)]
            R = K._돌림()
            xy = np.vstack([(V @ R.T)[:, [0, 2]] for V, _ in 셋])
            범위 = (xy.min(0), xy.max(0))
            y = 40 + k * (크기[1] + 30)
            d.text((10, y), c, fill="black", font=K._글꼴(14))
            for i, ((V, F), 색) in enumerate(zip(셋, [(200, 200, 200), (205, 175, 150), (150, 185, 215)])):
                im.paste(K.그리기({"몸": (V, F)}, 크기=크기, 색표={"몸": 색}, 범위=범위), (10 + i * 크기[0], y + 20))
            print(c, flush=True)
        im.save(os.path.join(A.OUT, "%s확대_%d.png" % (새이름, p // 3 + 1)))


if __name__ == "__main__":
    main()
