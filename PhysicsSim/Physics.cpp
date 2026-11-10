#include "Physics.h"
#include <algorithm>

namespace {
constexpr float kGravity = 9.81f;
constexpr float kRestitution = 0.35f;
constexpr float kFriction = 0.4f;
constexpr float kSlop = 0.005f;
constexpr float kCorrection = 0.8f;
// Cell size must be >= the largest centre distance at which two bodies can touch: two bounding spheres of
// radius 0.61 (a rotated cube's half-diagonal) = 1.22.
constexpr float kCell = 1.25f;
constexpr float kAngularDamping = 1.5f;   // 1/s, OBB mode only
constexpr float kMaxSpin = 25.0f;         // rad/s
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

// ---------------------------------------------------------------------------------------------
// OBB mode: oriented boxes + rigid-body rotation
// ---------------------------------------------------------------------------------------------

// Inverse inertia tensor applied to a world-space vector: R * diag(invI) * R^T * v
Vec3 InvInertia(const Body& b, Vec3 v) {
    return b.ax[0] * (Dot(b.ax[0], v) * b.invI.x) + b.ax[1] * (Dot(b.ax[1], v) * b.invI.y) +
           b.ax[2] * (Dot(b.ax[2], v) * b.invI.z);
}

void Orthonormalize(Body& b) {
    b.ax[0] = Normalize(b.ax[0]);
    b.ax[1] = Normalize(b.ax[1] - b.ax[0] * Dot(b.ax[1], b.ax[0]));
    b.ax[2] = Cross(b.ax[0], b.ax[1]);
}

void Rotate(Body& b, float h) {
    if (Dot(b.w, b.w) < 1e-12f) return;
    for (int k = 0; k < 3; ++k) b.ax[k] += Cross(b.w, b.ax[k]) * h;
    Orthonormalize(b);
}

void Corners(const Body& b, Vec3 out[8]) {
    for (int i = 0; i < 8; ++i)
        out[i] = b.pos + b.ax[0] * ((i & 1 ? 1.0f : -1.0f) * b.half.x) + b.ax[1] * ((i & 2 ? 1.0f : -1.0f) * b.half.y) +
                 b.ax[2] * ((i & 4 ? 1.0f : -1.0f) * b.half.z);
}

bool InsideObb(const Body& b, Vec3 p, float tol) {
    Vec3 r = p - b.pos;
    return std::fabs(Dot(r, b.ax[0])) <= b.half.x + tol && std::fabs(Dot(r, b.ax[1])) <= b.half.y + tol &&
           std::fabs(Dot(r, b.ax[2])) <= b.half.z + tol;
}

// A contact manifold: shared normal (a -> b) and up to 16 contact points.
struct Contact {
    Vec3 n;
    int count = 0;
    Vec3 p[16];
    float depth = 0;   // deepest penetration along n
};

// OBB vs OBB by the separating axis theorem (15 axes). Contact points are the corners of each box that lie
// inside the other one (face contacts give up to 4+ points); edge-edge contacts use the closest points of the edges.
bool ObbObb(const Body& a, const Body& b, Contact& c) {
    const float ha[3] = { a.half.x, a.half.y, a.half.z }, hb[3] = { b.half.x, b.half.y, b.half.z };
    const Vec3 d = b.pos - a.pos;

    float bestScore = 1e30f, bestPen = 0;
    Vec3 bestN;
    int type = 0, ia = 0, ib = 0;
    bool separated = false;

    auto test = [&](Vec3 L, int t, int i, int j) {
        float len = Length(L);
        if (len < 1e-4f) return;               // parallel edges: this axis adds nothing
        L = L * (1.0f / len);
        float ra = 0, rb = 0;
        for (int k = 0; k < 3; ++k) {
            ra += ha[k] * std::fabs(Dot(a.ax[k], L));
            rb += hb[k] * std::fabs(Dot(b.ax[k], L));
        }
        float dist = Dot(d, L);
        float pen = ra + rb - std::fabs(dist);
        if (pen <= 0) { separated = true; return; }
        float score = t == 2 ? pen * 1.05f + 0.002f : pen;    // prefer face contacts over edge contacts
        if (score < bestScore) {
            bestScore = score; bestPen = pen; type = t; ia = i; ib = j;
            bestN = dist >= 0 ? L : -L;
        }
    };

    for (int i = 0; i < 3 && !separated; ++i) test(a.ax[i], 0, i, 0);
    for (int i = 0; i < 3 && !separated; ++i) test(b.ax[i], 1, 0, i);
    for (int i = 0; i < 3 && !separated; ++i)
        for (int j = 0; j < 3 && !separated; ++j) test(Cross(a.ax[i], b.ax[j]), 2, i, j);
    if (separated) return false;

    const Vec3 n = bestN;
    c.n = n;
    c.depth = bestPen;
    c.count = 0;

    if (type == 2) {
        // edge-edge: closest points between the two supporting edges
        Vec3 pa = a.pos, pb = b.pos;
        for (int k = 0; k < 3; ++k) {
            if (k != ia) pa += a.ax[k] * ((Dot(a.ax[k], n) >= 0 ? 1.0f : -1.0f) * ha[k]);
            if (k != ib) pb += b.ax[k] * ((Dot(b.ax[k], n) >= 0 ? -1.0f : 1.0f) * hb[k]);
        }
        Vec3 u = a.ax[ia], v = b.ax[ib], r = pa - pb;
        float bb = Dot(u, v), cc = Dot(u, r), ff = Dot(v, r);
        float denom = 1.0f - bb * bb;
        float s = denom > 1e-8f ? (bb * ff - cc) / denom : 0.0f;
        s = Clampf(s, -ha[ia], ha[ia]);
        float t = Clampf(bb * s + ff, -hb[ib], hb[ib]);
        c.p[c.count++] = (pa + u * s + pb + v * t) * 0.5f;
        return true;
    }

    float ra = 0, rb = 0;
    for (int k = 0; k < 3; ++k) {
        ra += ha[k] * std::fabs(Dot(a.ax[k], n));
        rb += hb[k] * std::fabs(Dot(b.ax[k], n));
    }
    const float topA = Dot(a.pos, n) + ra;     // A's surface facing B
    const float botB = Dot(b.pos, n) - rb;     // B's surface facing A
    Vec3 corners[8];
    Corners(b, corners);
    for (const Vec3& v : corners)
        if (topA - Dot(v, n) > 0 && InsideObb(a, v, 0.02f)) c.p[c.count++] = v;
    Corners(a, corners);
    for (const Vec3& u : corners)
        if (Dot(u, n) - botB > 0 && InsideObb(b, u, 0.02f)) c.p[c.count++] = u;
    if (c.count == 0) c.p[c.count++] = (a.pos + n * ra + b.pos - n * rb) * 0.5f;
    return true;
}

// a = sphere, b = oriented box. n points from the sphere to the box.
bool SphereObb(const Body& a, const Body& b, Contact& c) {
    const float r = a.half.x, hb[3] = { b.half.x, b.half.y, b.half.z };
    Vec3 rel = a.pos - b.pos;
    float l[3], cl[3];
    for (int k = 0; k < 3; ++k) {
        l[k] = Dot(rel, b.ax[k]);
        cl[k] = Clampf(l[k], -hb[k], hb[k]);
    }
    Vec3 closest = b.pos + b.ax[0] * cl[0] + b.ax[1] * cl[1] + b.ax[2] * cl[2];
    Vec3 diff = closest - a.pos;
    float d2 = Dot(diff, diff);
    if (d2 > r * r) return false;
    c.count = 1;
    if (d2 > 1e-10f) {
        float dist = std::sqrt(d2);
        c.n = diff * (1.0f / dist);
        c.depth = r - dist;
        c.p[0] = closest;
        return true;
    }
    // centre inside the box: leave through the nearest face
    float best = 1e30f;
    int axis = 0;
    for (int k = 0; k < 3; ++k) {
        float pen = hb[k] - std::fabs(l[k]) + r;
        if (pen < best) { best = pen; axis = k; }
    }
    float sgn = l[axis] >= 0 ? 1.0f : -1.0f;
    c.n = b.ax[axis] * -sgn;
    c.depth = best;
    cl[axis] = sgn * hb[axis];
    c.p[0] = b.pos + b.ax[0] * cl[0] + b.ax[1] * cl[1] + b.ax[2] * cl[2];
    return true;
}

bool NarrowObb(const Body& a, const Body& b, Contact& c) {
    bool sa = a.shape == Shape::Sphere, sb = b.shape == Shape::Sphere;
    if (sa && sb) {
        Vec3 n;
        float depth;
        if (!SphereSphere(a, b, n, depth)) return false;
        c.n = n; c.depth = depth; c.count = 1;
        c.p[0] = a.pos + n * (a.half.x - depth * 0.5f);
        return true;
    }
    if (!sa && !sb) return ObbObb(a, b, c);
    if (sa) return SphereObb(a, b, c);
    if (!SphereObb(b, a, c)) return false;
    c.n = -c.n;
    return true;
}

// Sequential impulse at one contact point. n points from a to b; a or b may be static (invMass 0, invI 0).
void ContactImpulse(Body& a, Body& b, Vec3 n, Vec3 p) {
    const Vec3 ra = p - a.pos, rb = p - b.pos;
    auto relVel = [&] { return (b.vel + Cross(b.w, rb)) - (a.vel + Cross(a.w, ra)); };
    auto effMass = [&](Vec3 dir) {
        return a.invMass + b.invMass + Dot(Cross(InvInertia(a, Cross(ra, dir)), ra), dir) +
               Dot(Cross(InvInertia(b, Cross(rb, dir)), rb), dir);
    };
    auto apply = [&](Vec3 P) {
        a.vel -= P * a.invMass;
        a.w -= InvInertia(a, Cross(ra, P));
        b.vel += P * b.invMass;
        b.w += InvInertia(b, Cross(rb, P));
    };

    Vec3 rv = relVel();
    float vn = Dot(rv, n);
    if (vn >= 0) return;
    float e = -vn > 1.0f ? kRestitution : 0.0f;
    float jn = -(1 + e) * vn / effMass(n);
    apply(n * jn);

    rv = relVel();
    Vec3 t = rv - n * Dot(rv, n);
    float tl = Length(t);
    if (tl > 1e-6f) {
        t = t * (1.0f / tl);
        float jt = std::max(-tl / effMass(t), -kFriction * jn);
        apply(t * jt);
    }
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
    Vec3 inertia;   // principal moments of inertia (unit density scale, like mass)
    switch (s) {
    case Shape::Sphere: {
        float r = Rand(0.2f, 0.4f);
        b.half = {r, r, r};
        mass = 4.0f / 3.0f * kPi * r * r * r;
        float i = 0.4f * mass * r * r;
        inertia = {i, i, i};
        break;
    }
    case Shape::Cube: {
        b.half = {Rand(0.2f, 0.35f), Rand(0.2f, 0.35f), Rand(0.2f, 0.35f)};
        mass = 8 * b.half.x * b.half.y * b.half.z;
        Vec3 e = b.half * 2.0f;   // full edge lengths
        inertia = {mass * (e.y * e.y + e.z * e.z) / 12, mass * (e.x * e.x + e.z * e.z) / 12,
                   mass * (e.x * e.x + e.y * e.y) / 12};
        break;
    }
    case Shape::Cylinder:
    case Shape::Cone: {
        float r = Rand(0.15f, 0.3f), h = Rand(0.2f, 0.4f);
        b.half = {r, h, r};     // bounding volume (AABB / OBB) used for collision
        float len = 2 * h;
        if (s == Shape::Cylinder) {
            mass = kPi * r * r * len;
            inertia = {mass * (3 * r * r + len * len) / 12, 0.5f * mass * r * r, mass * (3 * r * r + len * len) / 12};
        } else {
            mass = kPi * r * r * len / 3.0f;
            float ixz = 3.0f / 20.0f * mass * r * r + 3.0f / 80.0f * mass * len * len;   // about the centroid
            inertia = {ixz, 0.3f * mass * r * r, ixz};
        }
        break;
    }
    }
    b.invMass = 1.0f / (mass * 3.0f);   // density 3 -> keeps masses in a comfortable range
    b.invI = {1.0f / (inertia.x * 3.0f), 1.0f / (inertia.y * 3.0f), 1.0f / (inertia.z * 3.0f)};
    if (obb_) b.w = {Rand(-3, 3), Rand(-3, 3), Rand(-3, 3)};
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
    for (Body& b : bodies_) {
        b.vel += Vec3(Rand(-6, 6), Rand(3, 10), Rand(-6, 6));
        if (obb_) b.w += Vec3(Rand(-8, 8), Rand(-8, 8), Rand(-8, 8));
    }
}

void World::SetObb(bool on) {
    if (on == obb_) return;
    obb_ = on;
    for (Body& b : bodies_) {
        if (on) {
            b.w = {Rand(-3, 3), Rand(-3, 3), Rand(-3, 3)};   // start spinning so the rotation is visible
        } else {
            b.ax[0] = {1, 0, 0}; b.ax[1] = {0, 1, 0}; b.ax[2] = {0, 0, 1};   // AABBs cannot rotate
            b.w = {};
        }
    }
}

// OBB mode walls: every box corner (or the lowest point of a sphere) that crosses a wall plane is a contact
// with an immovable body, so a box landing on an edge gets a torque and tips over.
void World::CollideWallsObb(Body& b) {
    const float lo[3] = {-W, 0, -D}, hi[3] = {W, H, D};
    Body wall;
    wall.invMass = 0;   // static: zero inverse mass and inertia
    for (int i = 0; i < 3; ++i)
        for (int side = 0; side < 2; ++side) {
            Vec3 n;
            SetComp(n, i, side == 0 ? 1.0f : -1.0f);   // points into the box
            const float plane = side == 0 ? lo[i] : hi[i];
            Vec3 pts[8];
            int count = 0;
            float maxDepth = 0;
            auto consider = [&](Vec3 p) {
                float depth = side == 0 ? plane - Comp(p, i) : Comp(p, i) - plane;
                if (depth <= 0) return;
                maxDepth = std::max(maxDepth, depth);
                pts[count++] = p;
            };
            if (b.shape == Shape::Sphere) {
                consider(b.pos - n * b.half.x);
            } else {
                Vec3 corners[8];
                Corners(b, corners);
                for (const Vec3& v : corners) consider(v);
            }
            if (count == 0) continue;
            b.pos += n * maxDepth;
            for (int k = 0; k < count; ++k) ContactImpulse(wall, b, n, pts[k] + n * maxDepth);
        }
}

void World::ResolveObb(Body& a, Body& b) {
    // cheap reject: bounding spheres (radius = half diagonal) must overlap before the 15-axis test
    Vec3 d = b.pos - a.pos;
    float r = Length(a.half) + Length(b.half);
    if (Dot(d, d) >= r * r) return;

    Contact c;
    if (!NarrowObb(a, b, c)) return;
    float sum = a.invMass + b.invMass;

    float corr = std::max(c.depth - kSlop, 0.0f) * kCorrection / sum;
    corr = std::min(corr, 0.5f);
    a.pos -= c.n * (corr * a.invMass);
    b.pos += c.n * (corr * b.invMass);

    for (int i = 0; i < c.count; ++i) ContactImpulse(a, b, c.n, c.p[i]);
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
        if (obb_) {
            b.w = b.w * std::max(0.0f, 1.0f - kAngularDamping * h);
            float spin = Length(b.w);
            if (spin > kMaxSpin) b.w = b.w * (kMaxSpin / spin);
            Rotate(b, h);
            CollideWallsObb(b);
        } else {
            CollideWalls(b);
        }
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
                                if (j > i) {
                                    if (obb_) ResolveObb(bodies_[i], bodies_[j]);
                                    else Resolve(bodies_[i], bodies_[j]);
                                }
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
        in.axisX = {b.ax[0].x, b.ax[0].y, b.ax[0].z, 0};
        in.axisY = {b.ax[1].x, b.ax[1].y, b.ax[1].z, 0};
        in.axisZ = {b.ax[2].x, b.ax[2].y, b.ax[2].z, 0};
        out[(int)b.shape].push_back(in);
    }
}
