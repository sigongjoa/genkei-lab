// 조각 엔진 C++ 판 — engine.py 와 같은 공식 · 같은 간격 · 같은 대칭 · 같은 잡기 (2026-09-22).
//
// 파이썬 판과 다른 것은 **계산 범위 하나**다. 파이썬은 자국마다 메시 전체의 법선과 공간 색인을 다시 만든다
// (21만 정점에서 자국당 144 ms 가 그 몫). 여기서는
//   공간 색인  균일 격자 · 획 시작 때 한 번 짓고, 움직인 정점만 칸을 옮긴다
//   법선       움직인 정점에 닿은 면만 다시 재고, 그 면의 정점만 다시 더한다
// 정답기는 engine.py 다 — 같은 붓질을 두 엔진에 넣어 정점 차이를 잰다 (engine_cpp.py 검사).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#define API extern "C" __declspec(dllexport)

namespace {

enum { 그리기 = 0, 부풀리기 = 1, 매끈 = 2, 잡기 = 3, 주름 = 4 };
const double 간격비 = 0.15;

struct V3 { double x, y, z; };
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double norm(V3 a) { return std::sqrt(dot(a, a)); }

inline double falloff(double d) {
    d = d < 0 ? 0 : (d > 1 ? 1 : d);
    return 3 * d * d * d * d - 4 * d * d * d + 1;
}

struct 잡힘 { std::vector<int> idx; std::vector<V3> v0; std::vector<double> f; double 거울; };

struct Engine {
    int n = 0, m = 0;
    std::vector<V3> V, V0, FN, VN;          // VN = 정규화한 정점 법선
    std::vector<int> F;
    std::vector<int> nbS, nb, vfS, vf;      // 이웃 정점 · 정점에 닿은 면 (CSR)
    // 격자
    double cell = 1;
    std::unordered_map<int64_t, std::vector<int>> grid;
    std::vector<int64_t> cellOf;
    // 획
    int 붓 = 0; double R = 1, I = 0.5; bool 대칭 = false, 반전 = false;
    bool 첫점 = true; V3 마지막{}; double 남은 = 0; int 자국수 = 0;
    bool 잡기준비 = false; V3 잡기시작{}; std::vector<잡힘> 잡;
    std::vector<char> mark;                  // 바뀐 정점 표시 (한 번의 점_더하기 동안)
    std::vector<int> 바뀐;

    int64_t key(V3 p) const {
        int64_t i = (int64_t)std::floor(p.x / cell), j = (int64_t)std::floor(p.y / cell), k = (int64_t)std::floor(p.z / cell);
        return (i & 0x1FFFFF) | ((j & 0x1FFFFF) << 21) | ((k & 0x1FFFFF) << 42);
    }
    void 격자짓기(double c) {
        cell = c; grid.clear(); cellOf.assign(n, 0);
        for (int i = 0; i < n; ++i) { cellOf[i] = key(V[i]); grid[cellOf[i]].push_back(i); }
    }
    void 격자옮김(int i) {
        int64_t k = key(V[i]);
        if (k == cellOf[i]) return;
        auto& a = grid[cellOf[i]]; a.erase(std::find(a.begin(), a.end(), i));
        grid[k].push_back(i); cellOf[i] = k;
    }
    // 반지름 안 (거리 <= R) · 번호 오름차순 — cKDTree.query_ball_point 와 같은 집합
    std::vector<int> 모으기(V3 c, double r) const {
        std::vector<int> out;
        int s = (int)std::ceil(r / cell);
        int64_t ci = (int64_t)std::floor(c.x / cell), cj = (int64_t)std::floor(c.y / cell), ck = (int64_t)std::floor(c.z / cell);
        double r2 = r * r;
        for (int64_t i = ci - s; i <= ci + s; ++i)
            for (int64_t j = cj - s; j <= cj + s; ++j)
                for (int64_t k = ck - s; k <= ck + s; ++k) {
                    auto it = grid.find((i & 0x1FFFFF) | ((j & 0x1FFFFF) << 21) | ((k & 0x1FFFFF) << 42));
                    if (it == grid.end()) continue;
                    for (int v : it->second) { V3 d = V[v] - c; if (dot(d, d) <= r2) out.push_back(v); }
                }
        std::sort(out.begin(), out.end());
        return out;
    }
    V3 면법선(int f) const {
        V3 a = V[F[3 * f]], b = V[F[3 * f + 1]], c = V[F[3 * f + 2]];
        return cross(b - a, c - a);
    }
    void 정점법선(int i) {
        V3 s{0, 0, 0};
        for (int t = vfS[i]; t < vfS[i + 1]; ++t) s = s + FN[vf[t]];
        double L = norm(s); VN[i] = L > 1e-12 ? s * (1.0 / L) : s * (1.0 / 1e-12);
    }
    void 전체법선() {
        for (int f = 0; f < m; ++f) FN[f] = 면법선(f);
        for (int i = 0; i < n; ++i) 정점법선(i);
    }
    void 국소갱신(const std::vector<int>& vs) {   // 움직인 정점 -> 닿은 면 -> 그 면의 정점
        std::vector<int> fs, ws;
        for (int v : vs) for (int t = vfS[v]; t < vfS[v + 1]; ++t) fs.push_back(vf[t]);
        std::sort(fs.begin(), fs.end()); fs.erase(std::unique(fs.begin(), fs.end()), fs.end());
        for (int f : fs) { FN[f] = 면법선(f); for (int k = 0; k < 3; ++k) ws.push_back(F[3 * f + k]); }
        std::sort(ws.begin(), ws.end()); ws.erase(std::unique(ws.begin(), ws.end()), ws.end());
        for (int w : ws) 정점법선(w);
        for (int v : vs) 격자옮김(v);
    }
    void 표시(int i) { if (!mark[i]) { mark[i] = 1; 바뀐.push_back(i); } }

    // 자국 하나 (대칭이면 거울 자국까지) — 같은 상태에서 다 계산하고 한꺼번에 더한다
    void 자국(V3 c) {
        std::vector<V3> 중심{c};
        if (대칭) 중심.push_back({-c.x, c.y, c.z});
        std::vector<std::vector<int>> 번호; std::vector<std::vector<V3>> 변위;
        double 부호 = 반전 ? -1.0 : 1.0;
        for (V3 q : 중심) {
            std::vector<int> idx = 모으기(q, R);
            std::vector<V3> dv(idx.size(), V3{0, 0, 0});
            if (!idx.empty()) {
                std::vector<double> f(idx.size());
                V3 nb_{0, 0, 0};
                for (size_t t = 0; t < idx.size(); ++t) { f[t] = falloff(norm(V[idx[t]] - q) / R); nb_ = nb_ + VN[idx[t]] * f[t]; }
                double L = norm(nb_); V3 nbar = nb_ * (1.0 / (L > 1e-12 ? L : 1e-12));
                for (size_t t = 0; t < idx.size(); ++t) {
                    int i = idx[t];
                    double ff = dot(VN[i], nbar) > 0 ? f[t] : 0.0;              // 뒷면 막기
                    V3 P = V[i];
                    switch (붓) {
                    case 그리기: dv[t] = nbar * (I * R * 0.1 * 부호 * ff); break;
                    case 부풀리기: dv[t] = VN[i] * (I * R * 0.1 * 부호 * ff); break;
                    case 매끈: {
                        V3 s{0, 0, 0}; int cnt = nbS[i + 1] - nbS[i];
                        for (int u = nbS[i]; u < nbS[i + 1]; ++u) s = s + V[nb[u]];
                        dv[t] = (s * (1.0 / cnt) - P) * (I * ff); break;
                    }
                    default: {  // 주름 — 기본이 판다
                        double f5 = ff * ff * ff * ff * ff;
                        dv[t] = (q - P) * (ff * 0.5 * I) - nbar * (I * R * 0.07 * 부호 * f5);
                    }
                    }
                }
            }
            번호.push_back(std::move(idx)); 변위.push_back(std::move(dv));
        }
        std::vector<int> 움직인;
        for (size_t s = 0; s < 번호.size(); ++s)
            for (size_t t = 0; t < 번호[s].size(); ++t) {
                V3 d = 변위[s][t]; int i = 번호[s][t];
                V[i] = V[i] + d;
                if (d.x != 0 || d.y != 0 || d.z != 0) { 표시(i); 움직인.push_back(i); }
            }
        std::sort(움직인.begin(), 움직인.end()); 움직인.erase(std::unique(움직인.begin(), 움직인.end()), 움직인.end());
        국소갱신(움직인);
        ++자국수;
    }

    void 잡기점(V3 p) {
        if (!잡기준비) {
            잡기준비 = true; 잡기시작 = p; 잡.clear();
            std::vector<std::pair<V3, double>> qs{{p, 1.0}};
            if (대칭) qs.push_back({V3{-p.x, p.y, p.z}, -1.0});
            for (auto& [q, 거울] : qs) {
                잡힘 g; g.idx = 모으기(q, R); g.거울 = 거울;
                for (int i : g.idx) { g.v0.push_back(V[i]); g.f.push_back(falloff(norm(V[i] - q) / R)); }
                잡.push_back(std::move(g));
            }
        }
        V3 δ = (p - 잡기시작) * I;
        std::vector<int> 움직인;
        for (auto& g : 잡) {
            V3 dd{δ.x * g.거울, δ.y, δ.z};
            for (size_t t = 0; t < g.idx.size(); ++t) { int i = g.idx[t]; V[i] = g.v0[t] + dd * g.f[t]; 표시(i); 움직인.push_back(i); }
        }
        std::sort(움직인.begin(), 움직인.end()); 움직인.erase(std::unique(움직인.begin(), 움직인.end()), 움직인.end());
        국소갱신(움직인);
        ++자국수;
    }

    void 점(V3 p) {
        if (붓 == 잡기) { 잡기점(p); return; }
        if (첫점) { 자국(p); 마지막 = p; 첫점 = false; return; }
        double 간 = 간격비 * R, L = norm(p - 마지막), t = 간 - 남은;
        V3 a = 마지막;
        while (L > 0 && t <= L) { 자국(a + (p - a) * (t / L)); t += 간; }
        남은 = L - (t - 간);
        마지막 = p;
    }
};

}  // namespace

API void* sc_create(const double* V, int n, const int* F, int m) {
    Engine* e = new Engine();
    e->n = n; e->m = m;
    e->V.resize(n); for (int i = 0; i < n; ++i) e->V[i] = {V[3 * i], V[3 * i + 1], V[3 * i + 2]};
    e->V0 = e->V;
    e->F.assign(F, F + 3 * m);
    // 이웃 — 면 모서리 양방향, 중복 없이, 번호 오름차순 (engine.py 의 np.unique 와 같은 순서)
    std::vector<std::vector<int>> nbl(n), vfl(n);
    for (int f = 0; f < m; ++f)
        for (int k = 0; k < 3; ++k) {
            int a = F[3 * f + k], b = F[3 * f + (k + 1) % 3];
            nbl[a].push_back(b); nbl[b].push_back(a); vfl[a].push_back(f);
        }
    e->nbS.assign(n + 1, 0); e->vfS.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        std::sort(nbl[i].begin(), nbl[i].end()); nbl[i].erase(std::unique(nbl[i].begin(), nbl[i].end()), nbl[i].end());
        e->nbS[i + 1] = e->nbS[i] + (int)nbl[i].size(); e->vfS[i + 1] = e->vfS[i] + (int)vfl[i].size();
    }
    for (int i = 0; i < n; ++i) { e->nb.insert(e->nb.end(), nbl[i].begin(), nbl[i].end()); e->vf.insert(e->vf.end(), vfl[i].begin(), vfl[i].end()); }
    e->FN.resize(m); e->VN.resize(n); e->mark.assign(n, 0);
    e->전체법선();
    return e;
}
API void sc_destroy(void* h) { delete (Engine*)h; }
API void sc_reset(void* h) { Engine* e = (Engine*)h; e->V = e->V0; e->전체법선(); }
API void sc_stroke_begin(void* h, int brush, double R, double I, int sym, int inv) {
    Engine* e = (Engine*)h;
    e->붓 = brush; e->R = R; e->I = I; e->대칭 = sym != 0; e->반전 = inv != 0;
    e->첫점 = true; e->남은 = 0; e->자국수 = 0; e->잡기준비 = false; e->잡.clear();
    e->격자짓기(R > 0 ? R : 1);
}
// 점 하나 -> 이번에 바뀐 정점 수. 번호는 out 에 (cap 까지), 오름차순
API int sc_add_point(void* h, double x, double y, double z, int* out, int cap) {
    Engine* e = (Engine*)h;
    e->바뀐.clear();
    e->점(V3{x, y, z});
    std::sort(e->바뀐.begin(), e->바뀐.end());
    int k = 0;
    for (int i : e->바뀐) { if (k < cap) out[k++] = i; e->mark[i] = 0; }
    return (int)e->바뀐.size();
}
API int sc_dab_count(void* h) { return ((Engine*)h)->자국수; }
API void sc_get_V(void* h, double* out) { Engine* e = (Engine*)h; std::memcpy(out, e->V.data(), sizeof(double) * 3 * e->n); }
API void sc_get_some(void* h, const int* idx, int k, double* out) {
    Engine* e = (Engine*)h;
    for (int t = 0; t < k; ++t) { V3 p = e->V[idx[t]]; out[3 * t] = p.x; out[3 * t + 1] = p.y; out[3 * t + 2] = p.z; }
}
