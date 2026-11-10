#include "AabbWorld.h"
#include <algorithm>
#include "PhysicsCommon.h"

using namespace phys;

namespace {
// Every body, spheres included, is treated as an axis-aligned box (a sphere's half-extents are {r, r, r}).
// Overlap on all three axes = collision; the axis with the smallest overlap is the contact normal.
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
}  // namespace

void AabbWorld::OnAdopt(Body& b) {
    b.ax[0] = {1, 0, 0};
    b.ax[1] = {0, 1, 0};
    b.ax[2] = {0, 0, 1};
    b.w = {};
}

void AabbWorld::UpdateBody(Body& b, float) { BounceOffWalls(b, b.half); }

void AabbWorld::Resolve(Body& a, Body& b) {
    Vec3 n;
    float depth;
    if (!BoxBox(a, b, n, depth)) return;
    float sum = a.invMass + b.invMass;

    // collision response: positional correction
    SeparateBodies(a, b, n, depth);

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
