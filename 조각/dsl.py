"""조형 DSL v0 — 기본형을 놓고 더하고 뺀다. **한 줄 = 저널 한 칸**, 다시 틀면 같은 해시.

    python 조각/dsl.py 검사

단위 mm · z 위 · 정면 -y (조각 앱 카메라 쪽) · 바닥 z0.
모든 기본형은 x, y = 바닥 가운데, z0 = 바닥 높이, rz = z 둘레 회전(도) 를 받는다.

    상자(w, d, h)          원기둥(r, h)          원뿔(r아래, r위, h)
    구(r)                  토러스(R, r, 축="z" | "x" | "y")

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


어휘 = {f.__name__: f for f in (상자, 원기둥, 원뿔, 구, 토러스)}


def _값(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, str)):
        return n.value
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return -_값(n.operand)
    raise ValueError("숫자나 문자열만 된다: %s" % ast.dump(n))


def _풀기(n):
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
        a, b = _풀기(n.left), _풀기(n.right)
        return a + b if isinstance(n.op, ast.Add) else a - b
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in 어휘:
        return 어휘[n.func.id](*[_값(a) for a in n.args], **{k.arg: _값(k.value) for k in n.keywords})
    raise ValueError("모르는 것: %s — 되는 것은 %s · + · -" % (ast.unparse(n), " · ".join(어휘)))


def 실행(줄):
    """DSL 한 줄 -> (V, F) 닫힌 다양체. 결정적이다."""
    man = _풀기(ast.parse(줄.strip(), mode="eval").body)
    assert not man.is_empty() and man.status() == m3.Error.NoError, "빈 모양이거나 다양체가 아니다: " + 줄
    me = man.to_mesh()
    return np.asarray(me.vert_properties, np.float64)[:, :3], np.asarray(me.tri_verts, np.int64)


def 해시(V, F):
    return hashlib.sha1(np.round(V, 3).tobytes() + np.asarray(F, np.int64).tobytes()).hexdigest()[:12]


def 검사():
    import trimesh
    줄들 = ["원기둥(r=12, h=150, x=-30) + 원기둥(r=12, h=150, x=30)",
          '토러스(R=48, r=24, 축="y")',
          "상자(w=90, d=90, h=150) - 상자(w=45, d=45, h=150, x=22.5, y=22.5)",
          "원뿔(r아래=52, r위=10, h=150) + 구(r=20, z0=135)"]
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
