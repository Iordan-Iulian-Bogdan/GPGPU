#include "Physics.h"
#include <algorithm>

namespace {
constexpr float kGravity = 9.81f;
constexpr float kRestitution = 0.35f;
constexpr float kFriction = 0.4f;
constexpr float kSlop = 0.005f;
constexpr float kCorrection = 0.8f;
constexpr float kCell = 1.0f;       // >= largest possible AABB extent sum (2 * 0.35 * 2 = 1.4 only on overlap axis: 0.7)
constexpr int kSubSteps = 2;
constexpr int NX = (int)(2 * World::W / kCell) + 1;
constexpr int NY = (int)(World::H / kCell) + 1;
constexpr int NZ = (int)(2 * World::D / kCell) + 1;
constexpr float kPi = 3.14159265f;

inline float Clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline int Clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float Comp(const Vec3& v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
inline void SetComp(Vec3& v, int i, float f) { (i == 0 ? v.x : (i == 1 ? v.y : v.z)) = f; }

// Contact between a and b. n points from a to b.
bool SphereSphere(const Body& a, const Body& b, Vec3& n, float& depth) {
    Vec3 d = b.pos - a.pos;
    float r = a.half.x + b.half.x, d2 = Dot(d, d);
    if (d2 >= r * r) return false;
    float dist = std::sqrt(d2);
    n = dist > 1e-6f ? d * (1.0f / dist) : Vec3(0, 1, 0);
    depth = r - dist;
    return true;
}

bool BoxBox(const Body& a, const Body& b, Vec3& n, float& depth) {
    Vec3 d = b.pos - a.pos;
    float best = 1e30f;
    int axis = 0;
    for (int i = 0; i < 3; ++i) {
        float o = Comp(a.half, i) + Comp(b.half, i) - std::fabs(Comp(d, i));
        if (o <= 0) return false;
        if (o < best) { best = o; axis = i; }
    }
    n = Vec3();
    SetComp(n, axis, Comp(d, axis) >= 0 ? 1.0f : -1.0f);
    depth = best;
    return true;
}

// a = sphere, b = AABB
bool SphereBox(const Body& a, const Body& b, Vec3& n, float& depth) {
    float r = a.half.x;
    Vec3 lo = b.pos - b.half, hi = b.pos + b.half;
    Vec3 c(Clampf(a.pos.x, lo.x, hi.x), Clampf(a.pos.y, lo.y, hi.y), Clampf(a.pos.z, lo.z, hi.z));
    Vec3 diff = c - a.pos;
    float d2 = Dot(diff, diff);
    if (d2 > r * r) return false;
    if (d2 > 1e-10f) {
        float dist = std::sqrt(d2);
        n = diff * (1.0f / dist);
        depth = r - dist;
        return true;
    }
    // sphere centre inside the box: leave through the nearest face
    float best = 1e30f;
    int axis = 0;
    Vec3 rel = a.pos - b.pos;
    for (int i = 0; i < 3; ++i) {
        float pen = Comp(b.half, i) - std::fabs(Comp(rel, i)) + r;
        if (pen < best) { best = pen; axis = i; }
    }
    n = Vec3();
    SetComp(n, axis, Comp(rel, axis) >= 0 ? -1.0f : 1.0f);
    depth = best;
    return true;
}

bool Narrow(const Body& a, const Body& b, Vec3& n, float& depth) {
    bool sa = a.shape == Shape::Sphere, sb = b.shape == Shape::Sphere;
    if (sa && sb) return SphereSphere(a, b, n, depth);
    if (!sa && !sb) return BoxBox(a, b, n, depth);
    if (sa) return SphereBox(a, b, n, depth);
    if (!SphereBox(b, a, n, depth)) return false;
    n = -n;
    return true;
}
}  // namespace

World::World() : rng_(12345) {
    bodies_.reserve(MaxBodies);
    head_.resize((size_t)NX * NY * NZ);
}

void World::Clear() { bodies_.clear(); }

void World::AddBody(Shape s, Vec3 pos, Vec3 vel) {
    if (bodies_.size() >= MaxBodies) return;
    Body b;
    b.shape = s;
    b.pos = pos;
    b.vel = vel;
    float mass = 1;
    switch (s) {
    case Shape::Sphere: {
        float r = Rand(0.2f, 0.4f);
        b.half = {r, r, r};
        mass = 4.0f / 3.0f * kPi * r * r * r;
        break;
    }
    case Shape::Cube:
        b.half = {Rand(0.2f, 0.35f), Rand(0.2f, 0.35f), Rand(0.2f, 0.35f)};
        mass = 8 * b.half.x * b.half.y * b.half.z;
        break;
    case Shape::Cylinder:
    case Shape::Cone: {
        float r = Rand(0.15f, 0.3f), h = Rand(0.2f, 0.4f);
        b.half = {r, h, r};     // bounding volume (AABB) used for collision
        mass = kPi * r * r * 2 * h * (s == Shape::Cone ? 1.0f / 3.0f : 1.0f);
        break;
    }
    }
    b.invMass = 1.0f / (mass * 3.0f);   // density 3 -> keeps masses in a comfortable range
    b.color = {Rand(0.25f, 1.0f), Rand(0.25f, 1.0f), Rand(0.25f, 1.0f)};
    bodies_.push_back(b);
}

void World::Spawn(Shape s, bool atTop) {
    (void)atTop;
    Vec3 p(Rand(-W + 0.6f, W - 0.6f), Rand(H - 6.0f, H - 1.0f), Rand(-D + 0.6f, D - 0.6f));
    Vec3 v(Rand(-4, 4), Rand(-4, 2), Rand(-4, 4));
    AddBody(s, p, v);
}

void World::LoadPreset(int index) {
    static const int counts[3][3] = {{100, 250, 500}, {250, 500, 1000}, {500, 1000, 2500}};
    index = Clampi(index, 0, 2);
    Clear();

    // jittered lattice so the random initial positions never overlap
    const float s = 0.9f;
    std::vector<Vec3> pts;
    for (float y = 0.5f; y < H - 0.5f; y += s)
        for (float x = -W + 0.45f; x < W - 0.45f; x += s)
            for (float z = -D + 0.45f; z < D - 0.45f; z += s) pts.push_back({x, y, z});
    std::shuffle(pts.begin(), pts.end(), rng_);

    std::vector<Shape> kinds;
    for (int i = 0; i < counts[index][0]; ++i) kinds.push_back(Shape::Sphere);
    for (int i = 0; i < counts[index][1]; ++i) kinds.push_back(Shape::Cube);
    for (int i = 0; i < counts[index][2]; ++i) kinds.push_back(i % 2 ? Shape::Cone : Shape::Cylinder);

    for (size_t i = 0; i < kinds.size() && i < pts.size(); ++i) {
        Vec3 p = pts[i] + Vec3(Rand(-0.05f, 0.05f), Rand(-0.05f, 0.05f), Rand(-0.05f, 0.05f));
        AddBody(kinds[i], p, Vec3(Rand(-3, 3), Rand(-3, 3), Rand(-3, 3)));
    }
}

void World::Kick() {
    for (Body& b : bodies_) b.vel += Vec3(Rand(-6, 6), Rand(3, 10), Rand(-6, 6));
}

void World::CollideWalls(Body& b) {
    const float lo[3] = {-W, 0, -D}, hi[3] = {W, H, D};
    for (int i = 0; i < 3; ++i) {
        for (int side = 0; side < 2; ++side) {
            float h = Comp(b.half, i);
            float p = Comp(b.pos, i), v = Comp(b.vel, i);
            float limit = side == 0 ? lo[i] + h : hi[i] - h;
            bool hit = side == 0 ? p < limit : p > limit;
            if (!hit) continue;
            SetComp(b.pos, i, limit);
            bool into = side == 0 ? v < 0 : v > 0;
            if (!into) continue;
            float vn = std::fabs(v);
            float e = vn > 1.0f ? kRestitution : 0.0f;
            SetComp(b.vel, i, -v * e);
            // Coulomb friction on the tangential components
            float jn = (1 + e) * vn;
            int t1 = (i + 1) % 3, t2 = (i + 2) % 3;
            float a = Comp(b.vel, t1), c = Comp(b.vel, t2);
            float tl = std::sqrt(a * a + c * c);
            if (tl > 1e-6f) {
                float f = std::min(tl, kFriction * jn) / tl;
                SetComp(b.vel, t1, a * (1 - f));
                SetComp(b.vel, t2, c * (1 - f));
            }
        }
    }
}

void World::Resolve(Body& a, Body& b) {
    Vec3 n;
    float depth;
    if (!Narrow(a, b, n, depth)) return;
    float sum = a.invMass + b.invMass;

    // collision response: positional correction
    float corr = std::max(depth - kSlop, 0.0f) * kCorrection / sum;
    corr = std::min(corr, 0.5f);
    a.pos -= n * (corr * a.invMass);
    b.pos += n * (corr * b.invMass);

    // collision response: impulse
    Vec3 rv = b.vel - a.vel;
    float vn = Dot(rv, n);
    if (vn >= 0) return;
    float e = -vn > 1.0f ? kRestitution : 0.0f;
    float jn = -(1 + e) * vn / sum;
    a.vel -= n * (jn * a.invMass);
    b.vel += n * (jn * b.invMass);

    // friction
    rv = b.vel - a.vel;
    Vec3 t = rv - n * Dot(rv, n);
    float tl = Length(t);
    if (tl > 1e-6f) {
        t = t * (1.0f / tl);
        float jt = std::max(-tl / sum, -kFriction * jn);
        a.vel -= t * (jt * a.invMass);
        b.vel += t * (jt * b.invMass);
    }
}

void World::SubStep(float h) {
    // gravity + integration + walls (cheap, serial)
    for (Body& b : bodies_) {
        b.vel.y -= kGravity * h;
        b.pos += b.vel * h;
        CollideWalls(b);
    }

    // broad phase: uniform grid (linked lists)
    std::fill(head_.begin(), head_.end(), -1);
    next_.assign(bodies_.size(), -1);
    auto cellOf = [](const Vec3& p, int& cx, int& cy, int& cz) {
        cx = Clampi((int)((p.x + World::W) / kCell), 0, NX - 1);
        cy = Clampi((int)(p.y / kCell), 0, NY - 1);
        cz = Clampi((int)((p.z + World::D) / kCell), 0, NZ - 1);
    };
    for (int i = 0; i < (int)bodies_.size(); ++i) {
        int cx, cy, cz;
        cellOf(bodies_[i].pos, cx, cy, cz);
        int c = (cz * NY + cy) * NX + cx;
        next_[i] = head_[c];
        head_[c] = i;
    }

    // narrow phase + response (same sequential solver as before, each pair once: j > i).
    // Parallelised by grid colouring: a body only touches bodies in the 3x3x3 cells around its own
    // cell, so two cells that are >= 3 apart on some axis never share a body. Cells are grouped by
    // (x%3, y%3, z%3) into 27 colours; the cells of one colour run in parallel, colours run one
    // after another. No locks or atomics needed, and the result does not depend on thread timing.
    for (int color = 0; color < 27; ++color) {
        const int ox = color % 3, oy = (color / 3) % 3, oz = color / 9;
        const int cxn = (NX - ox + 2) / 3, cyn = (NY - oy + 2) / 3, czn = (NZ - oz + 2) / 3;
        const int cells = cxn * cyn * czn;

        #pragma omp parallel for schedule(dynamic, 4) if (parallel_)
        for (int idx = 0; idx < cells; ++idx) {
            const int cx = ox + (idx % cxn) * 3;
            const int cy = oy + ((idx / cxn) % cyn) * 3;
            const int cz = oz + (idx / (cxn * cyn)) * 3;
            for (int i = head_[(cz * NY + cy) * NX + cx]; i >= 0; i = next_[i])
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            int x = cx + dx, y = cy + dy, z = cz + dz;
                            if (x < 0 || y < 0 || z < 0 || x >= NX || y >= NY || z >= NZ) continue;
                            for (int j = head_[(z * NY + y) * NX + x]; j >= 0; j = next_[j])
                                if (j > i) Resolve(bodies_[i], bodies_[j]);
                        }
        }
    }
}

void World::Step(float dt) {
    float h = dt / kSubSteps;
    for (int s = 0; s < kSubSteps; ++s) SubStep(h);
}

void World::BuildInstances(std::vector<Instance> out[4]) const {
    for (int i = 0; i < 4; ++i) out[i].clear();

    // the five visible walls (front face z=-D is omitted on purpose)
    const float t = 0.25f, gray[4] = {0.45f, 0.47f, 0.52f, 1};
    auto wall = [&](Vec3 c, Vec3 hs) {
        Instance in{{c.x, c.y, c.z, 0}, {hs.x, hs.y, hs.z, 0}, {gray[0], gray[1], gray[2], 1}};
        out[(int)Shape::Cube].push_back(in);
    };
    wall({0, -t, 0}, {W + 2 * t, t, D + 2 * t});               // floor
    wall({0, H / 2, D + t}, {W + 2 * t, H / 2, t});            // back
    wall({-W - t, H / 2, 0}, {t, H / 2, D});                   // left
    wall({W + t, H / 2, 0}, {t, H / 2, D});                    // right

    for (const Body& b : bodies_) {
        Instance in{{b.pos.x, b.pos.y, b.pos.z, 0}, {b.half.x, b.half.y, b.half.z, 0},
                    {b.color.x, b.color.y, b.color.z, 1}};
        out[(int)b.shape].push_back(in);
    }
}
