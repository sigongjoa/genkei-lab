"""키트 — 모양(DSL 한 줄)을 앞 그림의 부위 영역으로 잘라 부품으로 · 몸통과 닿는 곳에 끼움 핀 · 출력 판정 Q1~Q6 · 그림.

    python 조각/키트.py 검사

  자르기   앞 그림 부위 번호(그림.부위나누기)를 물체 밖까지 가장 가까운 부위로 채워 평면을 나눈다 → 부위마다 그 영역을
           깊이(y) 방향으로 밀어낸 기둥과 모양의 교집합 = 부품. 부품을 다 합치면 모양 그대로다(모양은 안 바뀐다).
  핀       몸통과 닿는 부위마다 경계 가운데에, 경계에 수직으로. 핀은 자식 부품에 붙고 몸통에는 구멍(공차 0.15 mm).
  판정     Q1 부품마다 닫힘 · Q2 접합면 반지름 >= 3 mm · Q3 0.8 mm 보다 얇은 부피 < 1% · Q4 부품 <= 15 ·
           Q5 부품이 다 이어짐 · Q6 자립(무게중심이 발바닥 볼록 껍질 안, 여유 >= 2 mm). Q6 이 떨어지면 받침을 더하고 다시 잰다.
           문턱은 원화3d/실험_20260921/조립성.py(키 150 mm · FDM 0.4 기준) 그대로, Q6 만 새로.
"""
import json
import os

import numpy as np
import manifold3d as m3
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage import measure

import dsl as D
import 그림 as G

M = m3.Manifold
공차 = 0.15
몸통 = 2
색 = {"머리": (231, 111, 81), "몸통": (170, 170, 180), "팔.왼": (86, 156, 214), "팔.오": (66, 130, 200),
     "다리.왼": (106, 190, 120), "다리.오": (86, 170, 100), "하체": (210, 175, 95), "받침": (120, 120, 120)}


def _글꼴(n):
    for p in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, n)
    return ImageFont.load_default()


# ── 자르기 ───────────────────────────────────────────────────────────────────

def 부위판(앞):
    """-> (평면 전체를 채운 부위 번호, 물체 안만 부위 번호)."""
    L = G.부위나누기(앞)
    L[(L == 0) & 앞] = 몸통                                            # 규칙이 못 정한 물체 화소는 몸통
    if not L.any():
        L[앞] = 몸통
    idx = ndimage.distance_transform_edt(L == 0, return_distances=False, return_indices=True)
    return L[idx[0], idx[1]], L


def _기둥(판, 번호, fa, Y):
    """부위 영역(앞 그림 평면)을 깊이 방향으로 [-Y, Y] 밀어낸 기둥."""
    m = np.pad(판 == 번호, 1).astype(float)
    polys = []
    for c in measure.find_contours(m, 0.5):
        r, col = c[:, 0] - 1, c[:, 1] - 1
        polys.append(np.stack([(col + 0.5 - fa.c) * fa.s, (fa.bot - r - 0.5) * fa.s], 1))
    cs = m3.CrossSection(polys, m3.FillRule.EvenOdd)
    return M.extrude(cs, 2 * Y).rotate([90, 0, 0]).translate([0, Y, 0])


# ── 핀 ───────────────────────────────────────────────────────────────────────

def 닿는곳(L):
    """물체 안 부위 경계 -> {(i, j): 경계 화소 (행, 열) 배열}."""
    out = {}
    for dr, dc in ((0, 1), (1, 0)):
        a, b = L[:L.shape[0] - dr, :L.shape[1] - dc], L[dr:, dc:]
        m = (a > 0) & (b > 0) & (a != b)
        rr, cc = np.nonzero(m)
        for i, j, r, c in zip(a[m], b[m], rr + dr / 2, cc + dc / 2):
            out.setdefault(tuple(sorted((int(i), int(j)))), []).append((r, c))
    return {k: np.array(v) for k, v in out.items()}


def 핀자리(k, P, L, 옆, fa, fo):
    """몸통(부모)과 자식 부위 경계 하나 -> 핀 한 개의 자리 · 방향 · 크기."""
    자식 = k[0] if k[1] == 몸통 else k[1]
    X, Z = (P[:, 1] + 0.5 - fa.c) * fa.s, (fa.bot - P[:, 0] - 0.5) * fa.s
    Q = np.stack([X, Z], 1)
    c = Q.mean(0)
    u, s, vt = np.linalg.svd(Q - c, full_matrices=False)
    t = vt[0] if len(Q) > 1 else np.array([1.0, 0.0])
    n = np.array([-t[1], t[0]])
    ys, xs = np.nonzero(L == 자식)
    자식중심 = np.array([(xs.mean() + 0.5 - fa.c) * fa.s, (fa.bot - ys.mean() - 0.5) * fa.s])
    if (자식중심 - c) @ n < 0:
        n = -n
    길이 = float(np.ptp(Q @ t)) + fa.s
    옆값 = G._옆줄(옆, fo, c[1]) or (0.0, 길이)
    y, 깊이 = 옆값
    작은 = min(길이, 깊이)
    return {"부모": G.부위이름[몸통], "자식": G.부위이름[자식], "자리": [round(float(c[0]), 2), round(float(y), 2), round(float(c[1]), 2)],
            "방향": [round(float(n[0]), 4), 0.0, round(float(n[1]), 4)], "r": round(float(np.clip(0.2 * 작은, 1.0, 4.0)), 2),
            "모양": 다보모양(G.부위이름[자식]),
            "반길이": round(float(np.clip(0.35 * 작은, 2.0, 6.0)), 2), "접합 반지름": round(작은 / 2, 2)}


# 다보(ダボ) 모양 — 원형사 규칙(09-23 조사): 자리 맞춤용이지 고정용이 아니다. 네모는 각도를 고정하고,
# 사다리꼴은 위아래를 알려 준다. 좌우로 닮은 부품은 모양을 다르게 해 바꿔 끼우지 못하게 한다.
# 둥근 다보는 돌아가고 좌우가 같아 둘 다 못 한다 — 09-23 까지 그것을 썼다.
def 다보모양(자식):
    """좌우 짝은 모양이 달라야 바꿔 끼우지 못한다. 네모는 90° 돌려도 들어가(4 겹 대칭) 각도를 한 자리로 못 잡는다 —
    그래서 한 자리만 맞는 두 모양을 쓴다: 사다리꼴(한쪽이 좁다) · 모따기네모(한 귀퉁이를 잘랐다)."""
    return "사다리꼴뒤집힘" if 자식.endswith(".오") else "사다리꼴"


def _다보로컬(종류, r, 반, 여유=0.0):
    """축이 +z 인 제자리 다보. 여유 > 0 이면 구멍(공차만큼 크고 길다)."""
    a, L = r + 여유, 반 + (0.3 if 여유 else 0.0)
    좁 = 0.5 * a                                                        # 한쪽이 좁다 — 위아래가 보이고 한 자리만 맞는다
    단면 = ((-a, -a), (a, -a), (좁, a), (-좁, a))
    if 종류 == "사다리꼴뒤집힘":                                          # 좁은 쪽이 반대 — 좌우를 바꿔 끼우면 안 들어간다
        단면 = tuple((x, -y) for x, y in 단면)
    return M.hull_points([[x, y, z] for z in (-L, L) for x, y in 단면])


def _다보(핀, 여유=0.0):
    θ = np.degrees(np.arctan2(핀["방향"][0], 핀["방향"][2]))
    return _다보로컬(핀["모양"], 핀["r"], 핀["반길이"], 여유).rotate([0, float(θ), 0]).translate(핀["자리"])


def 다보검사(핀들):
    """다보가 제 일을 하나 — (돌지 않나, 좌우가 바뀌나) 를 부피로 잰다.
    돌지 않음: 90° 돌린 다보가 제 구멍에 안 들어간다(밖으로 나온 부피 >= 5%).
    좌우 다름: 왼쪽 다보가 오른쪽 구멍에 안 들어간다."""
    def 밖(peg, hole):
        v = peg.volume()
        return float((peg - hole).volume() / v) if v > 0 else 0.0
    돌 = {}
    for p in 핀들:
        peg = _다보로컬(p["모양"], p["r"], p["반길이"])
        구멍 = _다보로컬(p["모양"], p["r"], p["반길이"], 공차)
        돌[p["자식"]] = round(밖(peg.rotate([0, 0, 90]), 구멍), 4)
    좌우 = {}
    for 왼 in [p for p in 핀들 if p["자식"].endswith(".왼")]:
        오 = next((q for q in 핀들 if q["자식"] == 왼["자식"][:-2] + ".오"), None)
        if 오:
            좌우[왼["자식"][:-2]] = round(밖(_다보로컬(왼["모양"], 왼["r"], 왼["반길이"]),
                                          _다보로컬(오["모양"], 오["r"], 오["반길이"], 공차)), 4)
    return 돌, 좌우


# ── 판정 ─────────────────────────────────────────────────────────────────────

def 복셀(V, F, 칸=0.27, 최대=2.5e7):
    """닫힌 메시 -> (bool 격자 [z, y, x], 쓴 칸 mm). manifold 를 높이마다 잘라 단면 다각형을 칠한다 — trimesh.voxelized 는
    큰 면(150 mm)을 칸까지 쪼개다 메모리가 바닥났다(09-23, cloak_Chest). 칸 수가 `최대` 를 넘으면 칸을 키운다."""
    from skimage.draw import polygon as 칠
    man = M(m3.Mesh(vert_properties=np.ascontiguousarray(V, np.float32), tri_verts=np.ascontiguousarray(F, np.uint32)))
    lo, hi = V.min(0), V.max(0)
    칸 = float(max(칸, (np.prod(hi - lo) / 최대) ** (1 / 3)))
    n = np.ceil((hi - lo) / 칸).astype(int) + 2
    g = np.zeros((n[2], n[1], n[0]), bool)
    for k in range(n[2]):
        z = lo[2] + (k - 0.5) * 칸
        for P in man.slice(z).to_polygons():
            P = np.asarray(P)
            rr, cc = 칠((P[:, 1] - lo[1]) / 칸 + 1, (P[:, 0] - lo[0]) / 칸 + 1, shape=g.shape[1:])
            g[k, rr, cc] ^= True                                           # 짝홀 — 구멍 윤곽은 다시 비운다
    return g, 칸


def 얇은몫(V, F, 칸=0.27, 최대=2.5e7):
    """0.8 mm 보다 얇은 부피의 몫 -> (몫, 쓴 칸 mm). 칸 복셀을 3×3×3 십자로 열었을 때 사라지는 몫."""
    g, 칸 = 복셀(V, F, 칸, 최대)
    if not g.any():
        return 0.0, 칸
    열린 = ndimage.binary_opening(g, structure=ndimage.generate_binary_structure(3, 1))
    return float(1 - 열린.sum() / g.sum()), 칸


def 빠짐(V, F, 칸=0.5):
    """틀에서 빠지는 축 -> ({"x"|"y"|"z": 걸린 칸 몫}, 격자, 걸린 칸). 축 방향 기둥마다 채워진 칸이 한 토막이면 그 축으로
    두 쪽 틀에서 빠진다(언더컷 없음) — art2real `조립.빠짐방향` 과 같은 뜻. 걸린 칸 = 두 토막 넘는 기둥의 칸."""
    g, _ = 복셀(V, F, 칸, 4e6)
    몫, 걸림 = {}, {}
    for 이름, ax in (("z", 0), ("y", 1), ("x", 2)):
        시작 = np.diff(np.concatenate([np.zeros_like(g.take([0], ax)), g], ax).astype(np.int8), axis=ax) == 1
        토막 = 시작.sum(ax, keepdims=True)
        나쁜 = g & (토막 > 1)
        몫[이름] = round(float(나쁜.sum() / max(g.sum(), 1)), 4)
        걸림[이름] = 나쁜
    return 몫, g, 걸림


def 자립(V, F):
    """-> (여유 mm, 무게중심 xy, 발바닥 점들). 여유 > 0 이면 무게중심이 발바닥 볼록 껍질 안."""
    m = trimesh.Trimesh(V, F, process=False)
    com = m.center_mass[:2]
    바닥 = V[V[:, 2] < V[:, 2].min() + 0.5][:, :2]
    if len(np.unique(np.round(바닥, 2), axis=0)) < 3:
        return -1e9, com, 바닥
    h = ConvexHull(바닥)
    여유 = -max(float(e[:2] @ com + e[2]) for e in h.equations)          # 면 방정식 n·x + d <= 0 이 안쪽
    return 여유, com, 바닥[h.vertices]


# ── 만들기 ───────────────────────────────────────────────────────────────────

def 만들기(앞, 옆, 키, 줄, 나누기=True, 다보=True):
    """-> (부품 {이름: (V, F)}, 핀 목록, 판정 dict, 받침 더함?). 나누기=False 면 한 덩어리(기본형은 한 부품이다)."""
    fa, fo = G.틀(앞, 키), G.틀(옆, 키)
    판, L = 부위판(앞)
    if not 나누기:
        판, L = np.full(판.shape, 몸통, np.uint8), np.where(앞, 몸통, 0).astype(np.uint8)
    모양 = D.매니폴드(줄)
    b = np.asarray(모양.bounding_box())
    Y = float(max(abs(b[1]), abs(b[4]))) + 5
    부품, 남은 = {}, 모양
    for 번호 in sorted(set(np.unique(L)) - {0, 몸통}):                 # 따로 딴 윤곽은 경계에서 조금 겹칠 수 있다 —
        조각 = 남은 ^ _기둥(판, 번호, fa, Y)                               # 남은 데서 떼어 가면 합이 모양과 꼭 같다
        if not 조각.is_empty() and 조각.volume() > 1.0:
            부품[G.부위이름[int(번호)]] = 조각
            남은 = 남은 - 조각
    부품["몸통"] = 남은
    핀들 = [핀자리(k, P, L, 옆, fa, fo) for k, P in sorted(닿는곳(L).items())
           if 몸통 in k and len(P) >= 5 and G.부위이름[k[0] if k[1] == 몸통 else k[1]] in 부품 and "몸통" in 부품]
    for 왼 in [p for p in 핀들 if p["자식"].endswith(".왼")]:      # 좌우 짝은 크기를 맞춘다 — 작은 다보는 큰 구멍에
        오 = next((q for q in 핀들 if q["자식"] == 왼["자식"][:-2] + ".오"), None)   # 헐렁하게 들어가 버린다(검들기 · monster)
        if 오:
            왼["r"] = 오["r"] = round(max(왼["r"], 오["r"]), 2)
            왼["반길이"] = 오["반길이"] = round(max(왼["반길이"], 오["반길이"]), 2)
    빠짐표 = {}                                                         # Q8 — 다보 달기 전 모양으로 잰다(구멍은 틀 뒤에 뚫는다)
    for k, v in 부품.items():
        몫 = 빠짐(*D.메시(v))[0]
        빠짐표[k] = [a for a, b in 몫.items() if b <= 0.002]
    for 핀 in (핀들 if 다보 else ()):
        부품[핀["자식"]] = 부품[핀["자식"]] + _다보(핀)
        부품["몸통"] = 부품["몸통"] - _다보(핀, 공차)
    메시 = {k: D.메시(v) for k, v in 부품.items() if not v.is_empty()}
    합 = D.메시(모양)
    여유, com, 발 = 자립(*합)
    받침 = 여유 < 2.0
    if 받침:                                                           # Q6 이 떨어지면 받침 — 무게중심 둘레 원판, 두께 4 mm
        반지름 = float(np.max(np.linalg.norm(발 - com, axis=1))) + 8 if len(발) else 30.0
        메시["받침"] = D.메시(M.cylinder(4.0, 반지름, 반지름, 64).translate([float(com[0]), float(com[1]), -4.0]))
    돌, 좌우 = 다보검사(핀들)
    얇칸 = {k: 얇은몫(*v) for k, v in 메시.items()}
    얇, 칸 = {k: v[0] for k, v in 얇칸.items()}, max(v[1] for v in 얇칸.values())
    이어짐 = {k for k in 메시 if k in ("몸통", "받침")} | {p["자식"] for p in 핀들}
    판정 = {
        "Q1 닫힘": {"통과": all(trimesh.Trimesh(*v, process=False).is_watertight for v in 메시.values()), "값": len(메시)},
        "Q2 접합면 반지름 >= 3 mm": {"통과": all(p["접합 반지름"] >= 3 for p in 핀들), "값": {p["자식"]: p["접합 반지름"] for p in 핀들}},
        "Q3 얇은 부피 < 1%": {"통과": all(v < 0.01 for v in 얇.values()), "값": {k: round(v, 4) for k, v in 얇.items()}, "칸 mm": round(칸, 3)},
        "Q4 부품 <= 15": {"통과": len(메시) <= 15, "값": len(메시)},
        "Q5 다 이어짐": {"통과": set(메시) <= 이어짐, "값": sorted(set(메시) - 이어짐)},
        "Q6 자립 여유 >= 2 mm": {"통과": True if 받침 else 여유 >= 2.0, "값": round(float(여유), 2), "받침": 받침},
        "Q8 틀에서 빠짐 (다보 전)": {"통과": all(빠짐표.values()), "값": {k: v or "못 빠짐" for k, v in 빠짐표.items()}},
        "Q7 다보 (돌지 않음 · 좌우 다름)": {"통과": all(v >= 0.05 for v in 돌.values()) and all(v >= 0.05 for v in 좌우.values()),
                                   "값": {"90도 돌리면 밖으로": 돌, "왼 다보 대 오른 구멍": 좌우}},
    }
    return 메시, 핀들, 판정, 받침


# ── 그림 ─────────────────────────────────────────────────────────────────────

def _돌림(yaw=35.0, pitch=18.0):
    a, b = np.radians(yaw), np.radians(pitch)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return Rx @ Rz


def 그리기(메시, 핀들=(), 벌림=0.0, 크기=(520, 640), 색표=None):
    """3/4 에서 본 그림 (화가 순서). 벌림 > 0 이면 부품을 핀 방향으로 벌림 mm 씩 빼낸 분해도."""
    R = _돌림()
    방향 = {p["자식"]: np.array(p["방향"]) for p in 핀들}
    tris, 색들 = [], []
    for k, (V, F) in 메시.items():
        off = 방향.get(k, np.array([0, 0, -1.0]) if k == "받침" else np.zeros(3)) * 벌림
        W = (V + off) @ R.T
        T = W[F]
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        앞면 = n[:, 1] < 0                                             # 카메라는 -y 에서 +y 를 본다
        빛 = 0.35 + 0.65 * np.clip(n[앞면] @ np.array([-0.35, -0.75, 0.55]) / np.linalg.norm([-0.35, -0.75, 0.55]), 0, 1)
        tris.append(T[앞면])
        색들.append(np.array((색표 or 색).get(k, (200, 200, 200)))[None] * 빛[:, None])
    T, C = np.concatenate(tris), np.concatenate(색들)
    xy = T[:, :, [0, 2]]
    lo, hi = xy.reshape(-1, 2).min(0), xy.reshape(-1, 2).max(0)
    s = 0.9 * min(크기[0] / (hi[0] - lo[0]), 크기[1] / (hi[1] - lo[1]))
    P = np.stack([(xy[..., 0] - (lo[0] + hi[0]) / 2) * s + 크기[0] / 2, 크기[1] / 2 - (xy[..., 1] - (lo[1] + hi[1]) / 2) * s], -1)
    im = Image.new("RGB", 크기, (250, 250, 252))
    d = ImageDraw.Draw(im)
    for i in np.argsort(-T[:, :, 1].mean(1)):                          # 먼 것부터
        d.polygon([tuple(p) for p in P[i]], fill=tuple(int(c) for c in C[i]))
    return im


def 시트(메시, 핀들, 판정, 제목):
    W, H = 1100, 900
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((24, 16), 제목, fill="black", font=_글꼴(24))
    im.paste(그리기(메시), (20, 60))
    im.paste(그리기(메시, 핀들, 25.0), (560, 60))
    d.text((40, 70), "조립", fill=(120, 120, 120), font=_글꼴(16))
    d.text((580, 70), "분해 · 핀 %d" % len(핀들), fill=(120, 120, 120), font=_글꼴(16))
    y = 715
    for k, v in 판정.items():
        값 = v["값"] if not isinstance(v["값"], dict) else ", ".join("%s %s" % kv for kv in list(v["값"].items())[:6])
        글 = "%s  %s  %s" % ("○" if v["통과"] else "×", k, 값)
        if k.startswith("Q6") and v.get("받침"):
            글 += "  → 받침을 더해 통과"
        d.text((30, y), 글[:110], fill=(40, 160, 80) if v["통과"] else (220, 60, 50), font=_글꼴(16))
        y += 28
    x = 30
    for k in 메시:
        d.rectangle([x, 695, x + 14, 709], fill=색.get(k, (200, 200, 200)))
        d.text((x + 18, 692), k, fill="black", font=_글꼴(14))
        x += 30 + 14 * len(k)
    return im


def 쓰기(메시, 핀들, 판정, 받침, 폴더, 제목, 줄):
    os.makedirs(os.path.join(폴더, "부품"), exist_ok=True)
    for k, (V, F) in 메시.items():
        trimesh.Trimesh(V, F, process=False).export(os.path.join(폴더, "부품", k + ".stl"))
    json.dump({"모양": 줄, "부품": {k: {"정점": int(len(V)), "부피 mm3": round(float(trimesh.Trimesh(V, F, process=False).volume), 1)}
                                  for k, (V, F) in 메시.items()},
               "핀": 핀들, "판정": 판정, "받침 더함": 받침, "공차 mm": 공차},
              open(os.path.join(폴더, "키트.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    시트(메시, 핀들, 판정, 제목).save(os.path.join(폴더, "키트.png"))


def 검사():
    """코드로 그린 사람 모양 — 부품 합 = 모양 · 핀 · 판정이 돈다."""
    im = Image.new("L", (400, 800), 255)
    d = ImageDraw.Draw(im)
    d.ellipse([160, 40, 240, 130], fill=0)                             # 머리
    d.rectangle([185, 125, 215, 150], fill=0)                          # 목
    d.rectangle([140, 150, 260, 420], fill=0)                          # 몸통
    d.rectangle([40, 160, 140, 195], fill=0); d.rectangle([260, 160, 360, 195], fill=0)     # 팔 (T)
    d.rectangle([145, 420, 195, 760], fill=0); d.rectangle([205, 420, 255, 760], fill=0)    # 다리
    d.rectangle([130, 740, 200, 760], fill=0); d.rectangle([200, 740, 270, 760], fill=0)    # 발
    앞 = np.asarray(im) < 128
    im2 = Image.new("L", (300, 800), 255)
    d2 = ImageDraw.Draw(im2)
    d2.ellipse([110, 40, 190, 130], fill=0); d2.rectangle([135, 125, 165, 150], fill=0)
    d2.rectangle([110, 150, 190, 420], fill=0); d2.rectangle([120, 420, 180, 760], fill=0); d2.rectangle([120, 740, 215, 760], fill=0)
    옆 = np.asarray(im2) < 128
    줄 = G.층타원(앞, 옆, 150.0)
    메시, 핀들, 판정, 받침 = 만들기(앞, 옆, 150.0, 줄)
    print("  부품", sorted(메시), "· 핀", [(p["자식"], p["r"]) for p in 핀들], "· 받침", 받침)
    for k, v in 판정.items():
        print("   ", "○" if v["통과"] else "✗", k, v["값"])
    합 = D.메시(D.매니폴드(줄))
    부피 = sum(trimesh.Trimesh(*v, process=False).volume for k, v in 메시.items() if k != "받침")
    print("  부품 부피 합 %.0f · 모양 부피 %.0f mm3" % (부피, trimesh.Trimesh(*합, process=False).volume))
    assert {"머리", "몸통", "팔.왼", "팔.오", "다리.왼", "다리.오"} <= set(메시), sorted(메시)
    assert len(핀들) >= 5, 핀들
    assert abs(부피 / trimesh.Trimesh(*합, process=False).volume - 1) < 0.01, "부품을 합치면 모양 부피 그대로 (핀 공차만큼만 빔)"
    assert 판정["Q1 닫힘"]["통과"] and 판정["Q4 부품 <= 15"]["통과"] and 판정["Q5 다 이어짐"]["통과"]
    시트(메시, 핀들, 판정, "키트 검사 — 코드로 그린 사람").save("키트_검사.png")
    print("키트 검사 통과 -> 키트_검사.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "검사":
        검사()
    else:
        print(__doc__)
