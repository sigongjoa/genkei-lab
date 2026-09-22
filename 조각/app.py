# -*- coding: utf-8 -*-
"""**조각 앱** — 붓질을 해 보고, 타임라인으로 되감고, 스크립트 재생과 결과가 같은지 확인한다 (2026-09-22).

    python app.py                 창을 띄운다
    python app.py 점검             창 없이 API 를 한 바퀴 돈다 (획 · 되감기 · 검증 · 저장)
    pyinstaller 조각.spec          -> dist/조각.exe

화면(ui/index.html · three.js)은 마우스를 **모델 위 세계 좌표**로 바꿔 아래 API 를 부른다.
붓 계산은 전부 `engine.py` — 저널을 스크립트로 재생해도 같은 함수를 지난다.
"""
import json
import os
import sys
import time

import numpy as np

import engine as E
try:                                   # C++ 엔진이 있으면 그것을 쓴다 (같은 결과 · 수백 배 빠르다 — engine_cpp.py 검사)
    import engine_cpp as EC
    엔진, 엔진이름 = EC.조각C, "C++"
except OSError:
    엔진, 엔진이름 = E.조각, "파이썬"

HERE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


class API:
    def __init__(self):
        self.기본 = {"종류": "구", "세분": 6}
        self._새엔진()

    # ------------------------------------------------------------ 메시
    def _새엔진(self, V=None, F=None):
        if V is None:
            V, F = E.기본메시(세분=self.기본.get("세분", 6))
        self.e = 엔진(V, F)
        self.전체 = []                    # 타임라인 전체 (되감아도 남는다)
        self.k = 0                        # 지금 보이는 단계

    def _메시(self):
        return {"V": np.round(self.e.V, 5).ravel().tolist(), "F": self.e.F.ravel().tolist(),
                "해시": self.e.해시(), "k": self.k, "n": len(self.전체), "엔진": 엔진이름}

    def 새구(self, 세분=6):
        self.기본 = {"종류": "구", "세분": int(세분)}
        self._새엔진()
        return self._메시()

    def 열기(self):
        import webview
        import trimesh
        r = webview.windows[0].create_file_dialog(webview.OPEN_DIALOG, file_types=("메시 (*.stl;*.ply;*.obj;*.glb)",))
        if not r:
            return None
        m = trimesh.load(r[0], force="mesh", process=True)
        V = np.asarray(m.vertices, np.float64)
        V = (V - (V.max(0) + V.min(0)) / 2) * (150.0 / max(np.ptp(V[:, 2]), 1e-9))   # 가운데 · 키 150 mm
        self.기본 = {"종류": "파일", "경로": r[0], "V": V.tolist(), "F": np.asarray(m.faces).tolist()}
        self._새엔진(V, np.asarray(m.faces))
        out = self._메시()
        out["알림"] = "정점 %d · 붓은 정점만 옮긴다 — 정점이 성기면 붓이 거칠게 먹는다" % len(V)
        return out

    # ------------------------------------------------------------ 획
    def 획_시작(self, 붓, 반지름, 세기, 대칭, 반전):
        if self.k < len(self.전체):                  # 과거에서 새로 칠하면 뒤는 버린다 (갈래 최소판)
            self.전체 = self.전체[:self.k]
        self.e.저널 = list(self.전체)
        self.e.획_시작(붓, 반지름, 세기, 대칭, 반전)
        self._t = time.time()
        return True

    def 점들(self, 점목록):
        바뀐 = set()
        for p in 점목록:
            바뀐 |= set(self.e.점_더하기(p))
        idx = sorted(바뀐)
        pos = self.e.위치(idx) if hasattr(self.e, "위치") else self.e.V[idx]
        return {"idx": idx, "pos": np.round(pos, 5).ravel().tolist()}

    def 획_끝(self):
        기록 = self.e.획_끝()
        if 기록 is None:
            return None
        self.전체.append(기록)
        self.k = len(self.전체)
        return {"k": self.k, "n": len(self.전체), "해시": 기록["해시"], "카드": self._카드(기록, self.k),
                "초": round(time.time() - self._t, 2)}

    def _카드(self, 기록, i):
        return {"i": i, "붓": 기록["붓"], "자국": 기록["자국수"], "점": len(기록["점"]), "반지름": 기록["반지름"],
                "세기": 기록["세기"], "대칭": 기록["대칭"], "반전": 기록["반전"], "해시": 기록["해시"]}

    def 카드들(self):
        return [self._카드(g, i + 1) for i, g in enumerate(self.전체)]

    # ------------------------------------------------------------ 타임라인
    def 되감기(self, k):
        k = int(max(0, min(k, len(self.전체))))
        if hasattr(self.e, "처음으로"):             # C++ — 엔진을 다시 짓지 않고 처음으로 돌려 다시 튼다
            self.e.처음으로(); self.e.저널 = []
            for 획 in self.전체[:k]:
                E.재생한획(self.e, 획)
        else:
            self.e = E.재생(self.전체[:k], self.e.V0, self.e.F)
        self.k = k
        return self._메시()

    def 검증(self):
        """지금 단계까지를 **새 엔진**으로 스크립트 재생 -> 화면에서 칠한 결과와 해시 비교."""
        t = time.time()
        r = 엔진(self.e.V0, self.e.F)                # 새 엔진 — 저널만 보고 처음부터
        for 획 in json.loads(json.dumps(self.전체[:self.k])):
            E.재생한획(r, 획)
        return {"화면": self.e.해시(), "스크립트": r.해시(), "같음": r.해시() == self.e.해시(),
                "단계": self.k, "초": round(time.time() - t, 2), "엔진": 엔진이름}

    # ------------------------------------------------------------ 파일
    def 저장(self):
        import webview
        r = webview.windows[0].create_file_dialog(webview.SAVE_DIALOG, save_filename="조각_저널.json")
        if not r:
            return None
        p = r if isinstance(r, str) else r[0]
        json.dump({"기본": self.기본 if self.기본["종류"] == "파일" else self.기본,
                   "저널": self.전체}, open(p, "w", encoding="utf-8"), ensure_ascii=False)
        return p

    def 저널열기(self):
        import webview
        r = webview.windows[0].create_file_dialog(webview.OPEN_DIALOG, file_types=("저널 (*.json)",))
        if not r:
            return None
        j = json.load(open(r[0], encoding="utf-8"))
        b = j.get("기본", {"종류": "구", "세분": 6})
        self.기본 = b
        if b["종류"] == "파일":
            self._새엔진(np.array(b["V"]), np.array(b["F"]))
        else:
            self._새엔진()
        self.전체 = j["저널"]
        return self.되감기(len(self.전체))

    def 내보내기(self):
        import webview
        import trimesh
        r = webview.windows[0].create_file_dialog(webview.SAVE_DIALOG, save_filename="조각.stl")
        if not r:
            return None
        p = r if isinstance(r, str) else r[0]
        trimesh.Trimesh(self.e.V, self.e.F, process=False).export(p)
        return p


def 점검():
    a = API()
    a.새구(4)
    선 = [[x, -49.0, 5.0] for x in np.linspace(-20, 20, 9)]
    for 붓, R, I, 대, 반 in [("그리기", 10, 0.5, False, False), ("부풀리기", 12, 0.4, True, False), ("주름", 8, 0.6, False, True)]:
        a.획_시작(붓, R, I, 대, 반)
        for i in range(0, len(선), 3):
            a.점들(선[i:i + 3])
        a.획_끝()
    a.획_시작("잡기", 15, 1.0, False, False); a.점들([[0, -50, 0], [0, -60, 4]]); a.획_끝()
    v = a.검증(); assert v["같음"], v
    h4 = a.e.해시()
    a.되감기(2); assert a.검증()["같음"]
    a.되감기(4); assert a.e.해시() == h4, "되감았다 돌아오면 같아야 한다"
    a.되감기(1); a.획_시작("매끈", 20, 0.5, False, False); a.점들(선); a.획_끝()
    assert len(a.전체) == 2 and a.검증()["같음"], "과거에서 칠하면 뒤를 버리고 새로"
    print("점검 통과 — 획 넷 · 검증 · 되감기 · 과거에서 새로 칠하기", v)


def main():
    import webview
    api = API()
    webview.create_window("조각 — 붓질 · 타임라인 · 재생 검증", os.path.join(HERE, "ui", "index.html"),
                          js_api=api, width=1440, height=900, min_size=(1000, 650), background_color="#F2F2F7")
    webview.start(http_server=True, debug="--debug" in sys.argv)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "점검":
        점검()
    else:
        main()
