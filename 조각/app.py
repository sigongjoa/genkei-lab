# -*- coding: utf-8 -*-
"""**조각 앱** — 붓질을 해 보고, 타임라인으로 되감고, 스크립트 재생과 결과가 같은지 확인한다 (2026-09-22).

    python app.py                 창을 띄운다
    python app.py 점검             창 없이 API 를 한 바퀴 돈다 (획 · 되감기 · 검증 · 저장)
    조각.exe 만들기 앞.png 옆.png 키mm 폴더   창 없이 그림 두 장 -> 폴더/{메시.stl, 저널.json, 결과.json}
    조각.exe 키트 앞.png 옆.png 키mm 폴더     위 + 부위로 잘라 핀 · 판정 -> 폴더/{부품/*.stl, 키트.json, 키트.png}
    pyinstaller 조각.spec          -> dist/조각.exe

화면(ui/index.html · three.js)은 마우스를 **모델 위 세계 좌표**로 바꿔 아래 API 를 부른다.
붓 계산은 전부 `engine.py` — 저널을 스크립트로 재생해도 같은 함수를 지난다.
"""
import json
import os
import sys
import time

import numpy as np

import dsl as D
import 그림 as 그
import engine as E
import remesh as RM
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
        self.V기본, self.F기본 = np.asarray(V, np.float64), np.asarray(F)
        self.e = 엔진(V, F)
        self.가지들 = [[]]                 # 가지마다 칸 목록 (붓 획 · 리메시). 과거에서 새로 칠하면 새 가지
        self.지금가지 = 0
        self.k = 0                        # 지금 보이는 단계

    @property
    def 전체(self):
        return self.가지들[self.지금가지]

    @전체.setter
    def 전체(self, v):
        self.가지들[self.지금가지] = v

    def _갈래(self):
        """과거 단계에서 새로 칠하려 하면 **새 가지**를 낸다 — 원래 가지는 그대로 남는다."""
        if self.k < len(self.전체):
            self.가지들.append(list(self.전체[:self.k]))
            self.지금가지 = len(self.가지들) - 1

    def 가지목록(self):
        return {"지금": self.지금가지, "가지": [{"i": i, "n": len(g)} for i, g in enumerate(self.가지들)]}

    def 가지고르기(self, i):
        self.지금가지 = int(i)
        return self.되감기(len(self.전체))

    def _메시(self):
        return {"V": np.round(self.e.V, 5).ravel().tolist(), "F": self.e.F.ravel().tolist(),
                "해시": self.e.해시(), "k": self.k, "n": len(self.전체), "엔진": 엔진이름,
                "가지": self.지금가지, "가지수": len(self.가지들)}

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

    # ------------------------------------------------------------ 재생 (붓 획 + 리메시)
    def _재생(self, 칸들):
        """처음 메시에서 칸들을 차례로 다시 튼다. 리메시 칸을 만나면 그 자리에서 메시를 새로 깔고 이어 간다."""
        e = 엔진(self.V기본, self.F기본)
        for g in 칸들:
            if g["붓"] == "리메시":
                V, F = RM.리메시(e.V, e.F, g["간격"])
                e = 엔진(V, F)
            elif g["붓"] == "DSL":
                e = 엔진(*D.실행(g["줄"]))
            else:
                E.재생한획(e, g)
        e.저널 = []
        return e

    def 리메시(self, 간격):
        self._갈래()
        t = time.time()
        V, F = RM.리메시(self.e.V, self.e.F, float(간격))
        self.e = 엔진(V, F)
        기록 = {"붓": "리메시", "간격": float(간격), "반지름": float(간격), "세기": 0.0, "대칭": False, "반전": False,
               "점": [], "자국수": 0, "정점": int(len(V)), "해시": self.e.해시()}
        self.전체.append(기록)
        self.k = len(self.전체)
        out = self._메시()
        out["알림"] = "리메시 %.2f mm — 정점 %d · %.1f초" % (float(간격), len(V), time.time() - t)
        return out

    def DSL(self, 줄, 덧=None):
        """DSL 한 줄로 메시를 새로 짓는다 (리메시처럼 한 칸). AI 는 붓 대신 이걸 부른다. 덧 = 저널에 같이 적을 것."""
        self._갈래()
        V, F = D.실행(줄)
        self.e = 엔진(V, F)
        기록 = {"붓": "DSL", "줄": 줄, "반지름": 0.0, "세기": 0.0, "대칭": False, "반전": False,
               "점": [], "자국수": 0, "정점": int(len(V)), "해시": self.e.해시(), **(덧 or {})}
        self.전체.append(기록)
        self.k = len(self.전체)
        return self._메시()

    def 그림에서(self, 앞경로, 옆경로, 키=150.0):
        """그림 두 장 -> 카탈로그 후보를 그림과 대 보고 1순위를 DSL 한 칸으로. 동점이면 사람이 고를 몫으로 남긴다."""
        후보, 동점 = 그.고르기(그.마스크(앞경로), 그.마스크(옆경로), float(키))
        self.후보, self.동점 = 후보, 동점
        out = self.DSL(후보[0]["줄"], {"출처": "그림", "그림": [os.path.basename(앞경로), os.path.basename(옆경로)], "키": float(키),
                                     "점수": 후보[0]["점수"], "동점": 동점})
        out["후보"], out["동점"] = 후보, 동점
        return out

    def 후보고르기(self, i):
        """「이상해요」에서 고른 후보 — 그것도 저널 한 칸."""
        h = self.후보[int(i)]
        return self.DSL(h["줄"], {"출처": "고름", "고른 후보": int(i), "점수": h["점수"]})

    # ------------------------------------------------------------ 획
    def 획_시작(self, 붓, 반지름, 세기, 대칭, 반전):
        self._갈래()
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
                "가지": self.지금가지, "가지수": len(self.가지들),
                "초": round(time.time() - self._t, 2)}

    def _카드(self, 기록, i):
        return {"i": i, "붓": 기록["붓"], "자국": 기록["자국수"], "점": len(기록["점"]), "반지름": 기록["반지름"],
                "세기": 기록["세기"], "대칭": 기록["대칭"], "반전": 기록["반전"], "해시": 기록["해시"],
                "간격": 기록.get("간격"), "정점": 기록.get("정점"), "줄": 기록.get("줄")}

    def 카드들(self):
        return [self._카드(g, i + 1) for i, g in enumerate(self.전체)]

    # ------------------------------------------------------------ 타임라인
    def 되감기(self, k):
        k = int(max(0, min(k, len(self.전체))))
        self.e = self._재생(self.전체[:k])           # ponytail: 매번 처음부터 — 단계가 많아 느려지면 스냅숏
        self.k = k
        return self._메시()

    def 검증(self):
        """지금 단계까지를 **새 엔진**으로 스크립트 재생 -> 화면에서 칠한 결과와 해시 비교."""
        t = time.time()
        r = self._재생(json.loads(json.dumps(self.전체[:self.k])))     # 새 엔진 — 저널만 보고 처음부터
        return {"화면": self.e.해시(), "스크립트": r.해시(), "같음": r.해시() == self.e.해시(),
                "단계": self.k, "초": round(time.time() - t, 2), "엔진": 엔진이름}

    # ------------------------------------------------------------ 파일
    def 저장(self):
        import webview
        r = webview.windows[0].create_file_dialog(webview.SAVE_DIALOG, save_filename="조각_저널.json")
        if not r:
            return None
        p = r if isinstance(r, str) else r[0]
        json.dump({"기본": self.기본, "저널": self.전체, "가지들": self.가지들, "지금가지": self.지금가지},
                  open(p, "w", encoding="utf-8"), ensure_ascii=False)
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
        self.가지들 = j.get("가지들") or [j["저널"]]
        self.지금가지 = min(j.get("지금가지", 0), len(self.가지들) - 1)
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
    assert len(a.전체) == 2 and a.검증()["같음"], "과거에서 칠하면 새 가지"
    assert len(a.가지들) == 2 and len(a.가지들[0]) == 4, "원래 가지는 네 칸 그대로 남는다"
    a.가지고르기(0); assert a.e.해시() == h4, "원래 가지로 돌아가면 원래 결과"
    a.가지고르기(1)
    a.리메시(1.5); a.획_시작("그리기", 10, 0.5, True, False); a.점들(선); a.획_끝()
    h = a.e.해시()
    assert len(a.전체) == 4 and a.검증()["같음"], "리메시 뒤 붓질까지 재생"
    a.되감기(2); a.되감기(4); assert a.e.해시() == h, "리메시를 건너 되감았다 돌아와도 같다"
    a.DSL('원기둥(r=12, h=150, x=-30) + 토러스(R=30, r=10, 축="y", x=30)')
    a.획_시작("부풀리기", 8, 0.4, False, False); a.점들([[-30, -12, 75], [-30, -12, 90]]); a.획_끝()
    h = a.e.해시()
    assert a.검증()["같음"], "DSL 뒤 붓질까지 재생"
    a.되감기(len(a.전체) - 2); a.되감기(len(a.전체)); assert a.e.해시() == h, "DSL 을 건너 되감았다 돌아와도 같다"
    print("점검 통과 — 획 넷 · 검증 · 되감기 · 갈래(원래 가지 보존) · 리메시 뒤 재생 · DSL 뒤 재생", v)


def 만들기(앞, 옆, 키, 폴더):
    """창 없이 그림 -> 형태. 벤치가 exe 를 이렇게 부른다 — 사람 버튼 · MCP 와 같은 `API.그림에서`."""
    import trimesh
    os.makedirs(폴더, exist_ok=True)
    a = API()
    r = a.그림에서(앞, 옆, float(키))
    trimesh.Trimesh(a.e.V, a.e.F, process=False).export(os.path.join(폴더, "메시.stl"))
    json.dump({"기본": a.기본, "저널": a.전체}, open(os.path.join(폴더, "저널.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({"해시": r["해시"], "엔진": 엔진이름, "후보": r["후보"], "동점": r["동점"], "exe": bool(getattr(sys, "frozen", False))},
              open(os.path.join(폴더, "결과.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def 키트(앞, 옆, 키, 폴더):
    """그림 두 장 -> 형태(만들기와 같다) -> 부위로 자른 출력 키트."""
    import 키트 as K
    만들기(앞, 옆, 키, 폴더)
    결과 = json.load(open(os.path.join(폴더, "결과.json"), encoding="utf-8"))
    줄 = 결과["후보"][0]["줄"]
    메시, 핀들, 판정, 받침 = K.만들기(그.마스크(앞), 그.마스크(옆), float(키), 줄)
    K.쓰기(메시, 핀들, 판정, 받침, 폴더, "%s — %s · 키 %s mm" % (os.path.basename(os.path.dirname(os.path.abspath(앞))), 결과["후보"][0]["이름"], 키), 줄)


def main():
    import webview
    api = API()
    webview.create_window("조각 — 붓질 · 타임라인 · 재생 검증", os.path.join(HERE, "ui", "index.html"),
                          js_api=api, width=1440, height=900, min_size=(1000, 650), background_color="#F2F2F7")
    webview.start(http_server=True, debug="--debug" in sys.argv)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "점검":
        점검()
    elif len(sys.argv) > 1 and sys.argv[1] in ("만들기", "키트"):
        try:
            (만들기 if sys.argv[1] == "만들기" else 키트)(*sys.argv[2:6])
        except Exception:                          # 창 없는 exe 는 stdout 이 없다 — 오류는 파일로
            import traceback
            os.makedirs(sys.argv[5], exist_ok=True)
            open(os.path.join(sys.argv[5], "오류.txt"), "w", encoding="utf-8").write(traceback.format_exc())
            sys.exit(1)
    else:
        main()
