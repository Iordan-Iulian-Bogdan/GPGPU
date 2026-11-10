#include "AabbWorld.h"
#include <algorithm>
#include "PhysicsCommon.h"

using namespace phys;

namespace {
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

void AabbWorld::OnAdopt(Body& b) {
    b.ax[0] = {1, 0, 0};
    b.ax[1] = {0, 1, 0};
    b.ax[2] = {0, 0, 1};
    b.w = {};
}

void AabbWorld::UpdateBody(Body& b, float) {
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

void AabbWorld::Resolve(Body& a, Body& b) {
    Vec3 n;
    float depth;
    if (!Narrow(a, b, n, depth)) return;
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
