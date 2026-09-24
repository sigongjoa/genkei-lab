"""조형 DSL v0 — 기본형을 놓고 더하고 뺀다. **한 줄 = 저널 한 칸**, 다시 틀면 같은 해시.

    python 조각/dsl.py 검사

단위 mm · z 위 · 정면 -y (조각 앱 카메라 쪽) · 바닥 z0.
모든 기본형은 x, y = 바닥 가운데, z0 = 바닥 높이, rz = z 둘레 회전(도) 를 받는다.

    상자(w, d, h)          원기둥(r, h)          원뿔(r아래, r위, h)
    구(r)                  토러스(R, r, 축="z" | "x" | "y")
    타원체(w, d, h)        캡슐(r, h)            반구(w, d, h)            둥근상자(w, d, h, r)
    각뿔대(w아래, w위, d아래, d위, h)     네모 단면이 곧게 변한다 — 사각뿔 · 쐐기 · 사다리꼴 기둥
    관(R, r안, 길이, 축="z" | "x" | "y")   속 빈 원기둥
    층쌓기(z=[...], x=[...], y=[...], w=[...], d=[...])   높이마다 **수평** 타원 단면(가운데 x · y, 폭 w, 깊이 d)을 잇는다.
        그림에서 높이마다 잰 폭 · 깊이를 그대로 넣는다 — 켜를 평평한 기둥으로 쌓던 계단(표면각 90°)이 없다.
    부드럽게(A + B + ..., 반경=3, 격자=1.0)   조각을 부드럽게 합친다(이음매를 반경 mm 로 둥글림 · 마칭 큐브).
    로프트(점=[[x,y,z], ...], w=[...], d=[...], 이름="팔.왼")   중심선을 따라 타원 단면을 잇는다.
        단면은 중심선에 수직 · d 는 깊이(y) 방향 지름, w 는 그에 수직인 앞 그림 쪽 지름. 이름은 부위(= 키트 부품) 표시.

    원기둥(r=12, h=150, x=-30) + 원기둥(r=12, h=150, x=30)      두 덩어리
    토러스(R=48, r=24, 축="y")                                   구멍 (축 y = 정면에서 구멍이 보인다)
    상자(w=90, d=90, h=150) - 상자(w=45, d=45, h=150, x=22.5, y=22.5)

읽는 법은 파이썬 식이지만 eval 하지 않는다 — 기본형 이름 · 숫자 · 문자열 · + · - 만 받는다.
"""
import ast
import hashlib

import numpy as np
import manifold3d as m3

M = m3.Manifold
둘레 = 96                                   # 원 한 바퀴 조각 수 (코드기하도형 과 같다)


def _놓기(man, x, y, z0, rz):
    if rz:
        man = man.rotate([0, 0, rz])
    return man.translate([x, y, z0])


def 상자(w, d, h, x=0, y=0, z0=0, rz=0):
    return _놓기(M.cube([w, d, h], True).translate([0, 0, h / 2]), x, y, z0, rz)


def 원기둥(r, h, x=0, y=0, z0=0, rz=0):
    return _놓기(M.cylinder(h, r, r, 둘레), x, y, z0, rz)


def 원뿔(r아래, r위, h, x=0, y=0, z0=0, rz=0):
    return _놓기(M.cylinder(h, r아래, r위, 둘레), x, y, z0, rz)


def 구(r, x=0, y=0, z0=0, rz=0):
    return _놓기(M.sphere(r, 둘레).translate([0, 0, r]), x, y, z0, rz)


def 토러스(R, r, 축="z", x=0, y=0, z0=0, rz=0):
    t = M.revolve(m3.CrossSection.circle(r, 32).translate([R, 0]), 64)          # 축 z
    if 축 == "y":
        t = t.rotate([90, 0, 0]).translate([0, 0, R + r])
    elif 축 == "x":
        t = t.rotate([0, 90, 0]).translate([0, 0, R + r])
    else:
        assert 축 == "z", 축
        t = t.translate([0, 0, r])
    return _놓기(t, x, y, z0, rz)


def 타원체(w, d, h, x=0, y=0, z0=0, rz=0):
    return _놓기(M.sphere(1, 둘레).scale([w / 2, d / 2, h / 2]).translate([0, 0, h / 2]), x, y, z0, rz)


def 캡슐(r, h, x=0, y=0, z0=0, rz=0):
    h = max(h, 2 * r)
    return _놓기((M.sphere(r, 둘레).translate([0, 0, r]) + M.sphere(r, 둘레).translate([0, 0, h - r])).hull(), x, y, z0, rz)


def 반구(w, d, h, x=0, y=0, z0=0, rz=0):
    구 = M.sphere(1, 둘레).scale([w / 2, d / 2, h])
    return _놓기(구 ^ M.cube([w + 2, d + 2, h + 1]).translate([-(w + 2) / 2, -(d + 2) / 2, 0]), x, y, z0, rz)


def 각뿔대(w아래, w위, d아래, d위, h, x=0, y=0, z0=0, rz=0):
    e = 1e-3                                                           # 꼭짓점이 한 점으로 모여도 hull 이 서게
    P = [[sx * max(w아래, e) / 2, sy * max(d아래, e) / 2, 0] for sx in (-1, 1) for sy in (-1, 1)] +         [[sx * max(w위, e) / 2, sy * max(d위, e) / 2, h] for sx in (-1, 1) for sy in (-1, 1)]
    return _놓기(M.hull_points(P), x, y, z0, rz)


def 둥근상자(w, d, h, r, x=0, y=0, z0=0, rz=0):
    r = min(r, w / 2, d / 2, h / 2) * 0.999
    구들 = [M.sphere(r, 둘레).translate([sx * (w / 2 - r), sy * (d / 2 - r), r + sz * (h - 2 * r)])
          for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)]
    return _놓기(M.batch_hull(구들), x, y, z0, rz)


def 관(R, r안, 길이, 축="z", x=0, y=0, z0=0, rz=0):
    t = M.cylinder(길이, R, R, 둘레) - M.cylinder(길이 + 2, r안, r안, 둘레).translate([0, 0, -1])
    if 축 == "y":
        t = t.translate([0, 0, -길이 / 2]).rotate([90, 0, 0]).translate([0, 0, R])
    elif 축 == "x":
        t = t.translate([0, 0, -길이 / 2]).rotate([0, 90, 0]).translate([0, 0, R])
    else:
        assert 축 == "z", 축
    return _놓기(t, x, y, z0, rz)


def _고리잇기(고리, 아래, 위):
    """고리 [n, 둘레, 3] + 두 끝 가운데 -> 닫힌 manifold (바깥을 보게)."""
    n, k = 고리.shape[0], 고리.shape[1]
    V = np.concatenate([고리.reshape(-1, 3), [아래, 위]])
    s, e, j = n * k, n * k + 1, np.arange(k)
    F = []
    for i in range(n - 1):
        a0, a1 = i * k + j, i * k + (j + 1) % k
        F += [np.stack([a0, a1, a1 + k], 1), np.stack([a0, a1 + k, a0 + k], 1)]
    F += [np.stack([np.full(k, s), (j + 1) % k, j], 1), np.stack([np.full(k, e), (n - 1) * k + j, (n - 1) * k + (j + 1) % k], 1)]
    F = np.concatenate(F)
    if np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() < 0:
        F = F[:, ::-1]
    return M(m3.Mesh(vert_properties=np.ascontiguousarray(V, np.float32), tri_verts=np.ascontiguousarray(F, np.uint32)))


def 층쌓기(z, x, y, w, d, 둘레=24, p=None):
    """p = 단면 꼴 지수(초타원) — 2 타원 · 클수록 네모. 없으면 타원(예전 줄 해시 그대로)."""
    z, x, y = (np.asarray(v, np.float64) for v in (z, x, y))
    w, d = np.maximum(np.asarray(w, np.float64), 0.2), np.maximum(np.asarray(d, np.float64), 0.2)
    assert len(z) >= 2 and len({len(z), len(x), len(y), len(w), len(d)}) == 1 and np.all(np.diff(z) > 0), "z 는 올라가고 칸 수가 같아야 한다"
    a = 2 * np.pi * np.arange(둘레) / 둘레
    c, s = np.cos(a)[None, :], np.sin(a)[None, :]
    if p is not None:
        e = 2 / np.asarray(p, np.float64)[:, None]
        assert e.shape[0] == len(z), "p 칸 수도 같아야 한다"
        c, s = np.sign(c) * np.abs(c) ** e, np.sign(s) * np.abs(s) ** e
    고리 = np.stack([x[:, None] + w[:, None] / 2 * c, y[:, None] + d[:, None] / 2 * s, np.repeat(z[:, None], 둘레, 1)], -1)
    return _고리잇기(고리, [x[0], y[0], z[0]], [x[-1], y[-1], z[-1]])


def 로프트(점, w, d, 이름="", 둘레=24):
    P = np.asarray(점, np.float64)
    w = np.maximum(np.asarray(w, np.float64), 0.2)
    d = np.maximum(np.asarray(d, np.float64), 0.2)
    n = len(P)
    assert n >= 2 and len(w) == n and len(d) == n, "점 · w · d 개수가 같아야 한다 (2 이상)"
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    Ew = np.cross([0.0, 1.0, 0.0], T)                                  # 앞 그림 안에서 중심선에 수직
    bad = np.linalg.norm(Ew, axis=1) < 1e-6                            # 중심선이 깊이 방향이면 x 를 쓴다
    Ew[bad] = [1.0, 0.0, 0.0]
    Ew /= np.linalg.norm(Ew, axis=1, keepdims=True)
    Ed = np.cross(T, Ew)
    a = 2 * np.pi * np.arange(둘레) / 둘레
    고리 = P[:, None] + (w[:, None, None] / 2) * np.cos(a)[None, :, None] * Ew[:, None]                      + (d[:, None, None] / 2) * np.sin(a)[None, :, None] * Ed[:, None]
    V = np.concatenate([고리.reshape(-1, 3), P[[0, -1]]])
    s, e, k = n * 둘레, n * 둘레 + 1, np.arange(둘레)
    F = []
    for i in range(n - 1):
        a0, a1 = i * 둘레 + k, i * 둘레 + (k + 1) % 둘레
        b0, b1 = a0 + 둘레, a1 + 둘레
        F += [np.stack([a0, a1, b1], 1), np.stack([a0, b1, b0], 1)]
    F += [np.stack([np.full(둘레, s), (k + 1) % 둘레, k], 1),
          np.stack([np.full(둘레, e), (n - 1) * 둘레 + k, (n - 1) * 둘레 + (k + 1) % 둘레], 1)]
    F = np.concatenate(F)
    부피 = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum()
    if 부피 < 0:
        F = F[:, ::-1]
    return M(m3.Mesh(vert_properties=np.ascontiguousarray(V, np.float32), tri_verts=np.ascontiguousarray(F, np.uint32)))


어휘 = {f.__name__: f for f in (상자, 원기둥, 원뿔, 구, 토러스, 타원체, 캡슐, 반구, 각뿔대, 둥근상자, 관, 로프트, 층쌓기)}


def _값(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, str)):
        return n.value
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return -_값(n.operand)
    if isinstance(n, (ast.List, ast.Tuple)):
        return [_값(e) for e in n.elts]
    raise ValueError("숫자나 문자열만 된다: %s" % ast.dump(n))


def _항들(n):
    """a + b + c -> [a, b, c] (부드럽게 안은 더하기만)."""
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
        return _항들(n.left) + _항들(n.right)
    return [n]


def 부드럽게(조각들, 반경=3.0, 격자=1.0, 흐림=1.0):
    """조각들을 **부드럽게 합친다** — 이음매(겨드랑이 · 가랑이 · 목)를 반경 mm 로 둥글린다.
    조각마다 높이마다 잘라 칠한 칸 -> 거리 변환 = 부호 거리장(속 -) · 다항식 부드러운 최솟값 · manifold level_set(마칭 사면체).
    모든 기본형이 된다(자른 면을 칠하니까). 격자 mm = 표면 세밀도. 결정적이다(같은 줄 -> 같은 해시)."""
    from scipy import ndimage
    from skimage.draw import polygon
    lo = np.min([m.bounding_box()[:3] for m in 조각들], 0) - 반경 - 3 * 격자
    hi = np.max([m.bounding_box()[3:] for m in 조각들], 0) + 반경 + 3 * 격자
    n = np.ceil((hi - lo) / 격자).astype(int) + 1
    D = np.full(n, np.inf)
    for m in 조각들:
        b0 = np.clip(np.floor((np.asarray(m.bounding_box()[:3]) - lo) / 격자).astype(int) - int(반경 / 격자) - 3, 0, n - 1)
        b1 = np.clip(np.ceil((np.asarray(m.bounding_box()[3:]) - lo) / 격자).astype(int) + int(반경 / 격자) + 3, 0, n - 1) + 1
        속 = np.zeros(b1 - b0, bool)
        for k in range(b0[2], b1[2]):
            z = lo[2] + k * 격자
            단 = m.slice(z).to_polygons()
            if not 단:
                continue
            칸 = np.zeros((b1[0] - b0[0], b1[1] - b0[1]), bool)
            for 고리 in 단:                                                  # 짝홀 — 구멍도 맞게 · 격자점이 안에 있는가만(치우침 없음)
                q = (np.asarray(고리) - lo[:2]) / 격자 - b0[:2]
                t = np.zeros_like(칸)
                t[polygon(q[:, 0], q[:, 1], 칸.shape)] = True
                칸 ^= t
            속[:, :, k - b0[2]] = 칸
        sd = (ndimage.distance_transform_edt(~속) - ndimage.distance_transform_edt(속)) * 격자
        blk = tuple(slice(a, b) for a, b in zip(b0, b1))
        a = D[blk]
        h = np.clip(0.5 + 0.5 * (a - sd) / 반경, 0, 1)                      # 다항식 smin (반경 = 섞는 폭)
        with np.errstate(invalid="ignore"):
            D[blk] = np.where(np.isinf(a), sd, a * (1 - h) + sd * h - 반경 * h * (1 - h))
    D[np.isinf(D)] = 10 * 반경
    if 흐림:                                                              # 칠한 칸의 거리는 격자 계단을 탄다 — 칸 σ 로 흐려 곡면으로(09-24 그림에서 봄)
        D = ndimage.gaussian_filter(D, 흐림)
    # manifold 의 level_set(체심 격자 · 마칭 사면체)은 늘 다양체를 낸다 — skimage 마칭 큐브는 얇은 곳에서 꼬였다(09-24)
    nx, ny, nz = D.shape

    def 거리(x, y, z):                                                    # 격자 값 삼선형 보간 · 속 = +
        u, v, w = (x - lo[0]) / 격자, (y - lo[1]) / 격자, (z - lo[2]) / 격자
        i, j, k = min(max(int(u), 0), nx - 2), min(max(int(v), 0), ny - 2), min(max(int(w), 0), nz - 2)
        fu, fv, fw = min(max(u - i, 0.0), 1.0), min(max(v - j, 0.0), 1.0), min(max(w - k, 0.0), 1.0)
        c = D[i:i + 2, j:j + 2, k:k + 2]
        c = c[0] * (1 - fu) + c[1] * fu
        c = c[0] * (1 - fv) + c[1] * fv
        return -float(c[0] * (1 - fw) + c[1] * fw)

    man = M.level_set(거리, list(lo) + list(lo + (np.asarray(D.shape) - 1) * 격자), 격자)
    # 면이 한 점에서 맞닿는 곳(스치는 두 다리)은 같은 좌표의 다른 정점으로 남는다 — manifold 는 되지만 STL 은 합쳐 구멍이 난다.
    # 그런 정점을 제 면 가운데 쪽으로 0.01 mm 떼어 놓는다(결정적).
    me = man.to_mesh()
    V = np.asarray(me.vert_properties, np.float64)[:, :3]
    F = np.asarray(me.tri_verts, np.int64)
    _, inv, cnt = np.unique(V, axis=0, return_inverse=True, return_counts=True)
    겹 = np.nonzero(cnt[inv.reshape(-1)] > 1)[0]
    if len(겹):
        중 = V[F].mean(1)
        for i in 겹:
            t = 중[(F == i).any(1)].mean(0) - V[i]
            V[i] += 0.01 * t / max(np.linalg.norm(t), 1e-12)
        man = M(m3.Mesh(vert_properties=np.ascontiguousarray(V, np.float32), tri_verts=np.ascontiguousarray(F, np.uint32)))
    return man


def _풀기(n):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "부드럽게":
        return 부드럽게([_풀기(t) for t in _항들(n.args[0])], **{k.arg: _값(k.value) for k in n.keywords})
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
        a, b = _풀기(n.left), _풀기(n.right)
        return a + b if isinstance(n.op, ast.Add) else a - b
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in 어휘:
        return 어휘[n.func.id](*[_값(a) for a in n.args], **{k.arg: _값(k.value) for k in n.keywords})
    raise ValueError("모르는 것: %s — 되는 것은 %s · + · -" % (ast.unparse(n), " · ".join(어휘)))


def 매니폴드(줄):
    """DSL 한 줄 -> manifold 객체 (불리언을 더 할 때 — 키트 자르기)."""
    man = _풀기(ast.parse(줄.strip(), mode="eval").body)
    assert not man.is_empty() and man.status() == m3.Error.NoError, "빈 모양이거나 다양체가 아니다: " + 줄
    return man


def 메시(man):
    """manifold -> (V, F) 정렬된 순서."""
    me = man.to_mesh()
    return 정렬(np.asarray(me.vert_properties, np.float64)[:, :3], np.asarray(me.tri_verts, np.int64))


def 실행(줄):
    """DSL 한 줄 -> (V, F) 닫힌 다양체. 결정적이다."""
    man = 매니폴드(줄)
    me = man.to_mesh()
    return 정렬(np.asarray(me.vert_properties, np.float64)[:, :3], np.asarray(me.tri_verts, np.int64))


def 정렬(V, F):
    """정점 · 면 순서를 한 가지로 — hull 은 모양은 같아도 순서가 매번 달라(병렬) 해시가 흔들린다."""
    V = np.round(V, 4)
    o = np.lexsort(V.T[::-1])
    inv = np.empty_like(o)
    inv[o] = np.arange(len(o))
    V, F = V[o], inv[F]
    k = np.argmin(F, 1)                                                # 면마다 가장 작은 번호가 앞 (돌림 방향은 그대로)
    F = np.take_along_axis(F, (k[:, None] + np.arange(3)) % 3, 1)
    return V, F[np.lexsort(F.T[::-1])]


def 해시(V, F):
    return hashlib.sha1(np.round(V, 3).tobytes() + np.asarray(F, np.int64).tobytes()).hexdigest()[:12]


def 검사():
    import trimesh
    줄들 = ["원기둥(r=12, h=150, x=-30) + 원기둥(r=12, h=150, x=30)",
          '토러스(R=48, r=24, 축="y")',
          "상자(w=90, d=90, h=150) - 상자(w=45, d=45, h=150, x=22.5, y=22.5)",
          "원뿔(r아래=52, r위=10, h=150) + 구(r=20, z0=135)",
          "타원체(w=60, d=40, h=150) + 캡슐(r=10, h=80, x=50)",
          '반구(w=100, d=80, h=40) + 각뿔대(w아래=60, w위=0, d아래=60, d위=0, h=90, z0=40) + 관(R=20, r안=12, 길이=30, 축="y", x=70)',
          "둥근상자(w=80, d=60, h=150, r=10)",
          '로프트(점=[[0,0,0],[0,0,60],[0,0,120]], w=[40,20,30], d=[30,15,20], 이름="몸통") + 로프트(점=[[15,0,100],[60,0,110]], w=[10,8], d=[10,8], 이름="팔.왼")']
    for 줄 in 줄들:
        V, F = 실행(줄)
        m = trimesh.Trimesh(V, F, process=False)
        assert m.is_watertight and m.volume > 0, 줄
        assert 해시(*실행(줄)) == 해시(V, F), "같은 줄은 같은 메시"
        print("  %-70s 정점 %5d · 덩어리 %d · 오일러 %d · %s" % (줄, len(V), len(m.split(only_watertight=False)), m.euler_number, 해시(V, F)))
    m = trimesh.Trimesh(*실행(줄들[0]), process=False)
    assert len(m.split(only_watertight=False)) == 2, "두 덩어리"
    assert trimesh.Trimesh(*실행(줄들[1]), process=False).euler_number == 0, "구멍 하나 (토러스 오일러 0)"
    for 나쁜 in ["__import__('os')", "상자(w=1, d=1, h=1) * 2", "open('x')"]:
        try:
            실행(나쁜)
            raise AssertionError("받으면 안 된다: " + 나쁜)
        except ValueError:
            pass
    print("dsl 검사 통과 — 닫힘 · 결정성 · 두 덩어리 · 구멍 · 모르는 식 거절")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "검사":
        검사()
    else:
        print(__doc__)
