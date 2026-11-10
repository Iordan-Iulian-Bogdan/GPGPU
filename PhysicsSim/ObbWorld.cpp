#include "ObbWorld.h"
#include <algorithm>
#include "PhysicsCommon.h"

using namespace phys;

namespace {
constexpr float kAngularDamping = 1.5f;   // 1/s
constexpr float kMaxSpin = 25.0f;         // rad/s

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

bool Narrow(const Body& a, const Body& b, Contact& c) {
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

void ObbWorld::InitBody(Body& b) { b.w = {Rand(-3, 3), Rand(-3, 3), Rand(-3, 3)}; }

void ObbWorld::OnKick(Body& b) { b.w += Vec3(Rand(-8, 8), Rand(-8, 8), Rand(-8, 8)); }

void ObbWorld::OnAdopt(Body& b) { b.w = {Rand(-3, 3), Rand(-3, 3), Rand(-3, 3)}; }   // visible spin right away

// Walls: every box corner (or the lowest point of a sphere) that crosses a wall plane is a contact with an
// immovable body, so a box landing on an edge gets a torque and tips over.
void ObbWorld::UpdateBody(Body& b, float h) {
    b.w = b.w * std::max(0.0f, 1.0f - kAngularDamping * h);
    float spin = Length(b.w);
    if (spin > kMaxSpin) b.w = b.w * (kMaxSpin / spin);
    Rotate(b, h);

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

void ObbWorld::Resolve(Body& a, Body& b) {
    // cheap reject: bounding spheres (radius = half diagonal) must overlap before the 15-axis test
    Vec3 d = b.pos - a.pos;
    float r = Length(a.half) + Length(b.half);
    if (Dot(d, d) >= r * r) return;

    Contact c;
    if (!Narrow(a, b, c)) return;
    SeparateBodies(a, b, c.n, c.depth);
    for (int i = 0; i < c.count; ++i) ContactImpulse(a, b, c.n, c.p[i]);
}
