# genkei-lab — 피규어 원형(原型) 연구실

원화에서 **출력 키트**(부품 STL · 분해도 · 색별 출력)까지 가는 피규어 제작 프로그램을 만들며 배우는 저장소.
원칙은 셋이다.

- **도면이 중심이다.** 그림에는 수치가 없다. 사람이 고치는 것은 수치 도면이고, 3D 는 도면에서 계산된다.
- **GUI 와 스크립트는 같은 명령을 부른다.** 조작마다 명령 한 줄이 저널에 남고, 저널을 다시 틀면 같은 결과가 나와야 한다.
- **직접 지은 것만 믿는다.** 핵심 연산은 우리 코드로 짓고, 옮길 때는 옛 판을 정답기로 삼아 결과를 대 본다.

## 조각 (`조각/`) — 스컬프트 최소판

붓 다섯(그리기 · 부풀리기 · 매끈 · 잡기 · 주름) · 대칭 X · 반대로(파기) · 리메시 · 타임라인(되감기 · 타임랩스 · 갈래) · 스크립트 재생 검증.

| 파일 | 하는 일 |
|---|---|
| `engine.py` | 붓 계산 파이썬 판 — **정답기**. 붓 공식은 SculptGL 과 같다 |
| `cpp/sculpt.cpp` | 같은 계산의 C++ 판(DLL). 붓 닿은 곳만 계산해 21만 정점에서 자국당 0.6 ms |
| `engine_cpp.py` | C++ 판을 파이썬에서 부르고, 정답기와 정점 차이를 잰다 |
| `dsl.py` | 조형 DSL v0 — 상자 · 원기둥 · 원뿔 · 구 · 토러스를 + · - 로. 한 줄 = 저널 한 칸 (`API.DSL`) |
| `remesh.py` | 복셀 거리장 + 마칭 큐브로 고르게 다시 깔기 |
| `app.py` · `ui/` | 창(pywebview + three.js). 마우스를 **모델 위 좌표(mm)** 로 바꿔 엔진을 부른다 |

### 돌리기

```
pip install numpy scipy trimesh scikit-image pywebview
python 조각/app.py              # 창
python 조각/engine.py 검사       # 결정성 · 재생 · 대칭 · 매끈 · 되감기
python 조각/engine_cpp.py        # 파이썬 판 = C++ 판 (최대 차이) · 속도
python 조각/remesh.py            # 리메시 결정성 · 모서리 고름 · 모양 유지
python 조각/app.py 점검          # 창 없이 API 한 바퀴 (갈래 · 리메시 뒤 재생 포함)
```

C++ DLL 다시 굽기 (MSVC):

```
cl /nologo /O2 /std:c++17 /utf-8 /EHsc /LD 조각/cpp/sculpt.cpp /Fe:조각/cpp/sculpt.dll
```

exe (PyInstaller):

```
cd 조각
pyinstaller --onefile --windowed --name 조각 --add-data "ui;ui" --add-binary "cpp/sculpt.dll;cpp" --collect-submodules trimesh --collect-submodules skimage.measure --hidden-import scipy.spatial app.py
```

## 벤치 (`벤치/`)

조각 메시(mm)를 art2real `원화3d` 벤치(도형 11 · 마네킹 7 · 실물 28)의 자로 잰다. 채점 코드와 정답은 `원화3d` 에 있어 공개 저장소만으로는 안 돈다.

| 파일 | 하는 일 |
|---|---|
| `어댑터.py` | 좌표 맞추기(조각 정면 -y → 벤치 정면 +y · 키 1 로 앉힘) · 3D IoU(속 채움 두 규약) · 앞 · 옆 실루엣 IoU |
| `도형DSL.py` | DSL v0: 시트 그림 두 장에서 잰 수치로 기본형을 놓는다. 식별 가능 여덟 중앙 0.997 · 토러스 1.000 · 못 가르는 셋은 후보를 고른다 |
| `대조군.py` | 도형 11 바닥 두 줄: 구 그대로(중앙 0.360) · 수축포장(정점만 옮긴 끝). 예측 먼저 적고 잰다 |

```
python 벤치/어댑터.py 검사     # 정답 왕복 12 장 = 1.000 · 축을 틀리면 떨어진다 -> 벤치/out/어댑터_검사.png
python 벤치/대조군.py          # -> 벤치/out/대조군.{json,png}
python 벤치/도형DSL.py         # -> 벤치/out/도형DSL.{json,png} (대조군을 먼저)
python 조각/dsl.py 검사         # DSL 닫힘 · 결정성 · 두 덩어리 · 구멍 · 모르는 식 거절
```

## 외부 코드

- [three.js](https://threejs.org) r160 — MIT (`조각/ui/lib/LICENSE-three.txt`)
- 붓 공식은 [SculptGL](https://github.com/stephomi/sculptgl)(MIT)의 식을 따라 새로 썼다
