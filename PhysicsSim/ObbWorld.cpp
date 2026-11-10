// Simplest OBB model: 15-axis SAT, one contact point per pair, walls treated as axis-aligned.
// Cones are approximated by three spheres along their axis; every other shape collides as an oriented box.
#include "ObbWorld.h"
#include <algorithm>
#include "PhysicsCommon.h"

using namespace phys;

namespace {
constexpr float kAngularDamping = 1.5f;   // 1/s (walls give no torque, so spin needs some other way to die out)

// Inverse inertia tensor applied to a world-space vector: R * diag(invI) * R^T * v
Vec3 InvInertia(const Body& b, Vec3 v) {
    return b.ax[0] * (Dot(b.ax[0], v) * b.invI.x) + b.ax[1] * (Dot(b.ax[1], v) * b.invI.y) +
           b.ax[2] * (Dot(b.ax[2], v) * b.invI.z);
}

// Turns the body by its angular velocity, then repairs the axes so they stay perpendicular and unit length.
void Rotate(Body& b, float h) {
    for (int k = 0; k < 3; ++k) b.ax[k] += Cross(b.w, b.ax[k]) * h;
    b.ax[0] = Normalize(b.ax[0]);
    b.ax[1] = Normalize(b.ax[1] - b.ax[0] * Dot(b.ax[1], b.ax[0]));
    b.ax[2] = Cross(b.ax[0], b.ax[1]);
}

// A cone is approximated by three spheres on its axis, each fitting inside the cone (base at local y = -h,
// apex at y = +h). The first is the largest sphere touching both the base and the side, the others shrink
// towards the tip. c = centre height along the axis, s = radius.
struct ConeSpheres {
    float c[3], s[3];
};

ConeSpheres MakeConeSpheres(const Body& b) {
    const float r = b.half.x, h = b.half.y, L = std::sqrt(r * r + 4 * h * h);
    ConeSpheres cs;
    cs.s[0] = 2 * h * r / (L + r);
    cs.c[0] = -h + cs.s[0];
    cs.c[1] = 0.0f;
    cs.s[1] = r * (h - cs.c[1]) / L;
    cs.c[2] = 0.45f * h;
    cs.s[2] = r * (h - cs.c[2]) / L;
    return cs;
}

// Extent of the body along each world axis (what its AABB would be).
Vec3 WorldExtent(const Body& b) {
    Vec3 e;
    if (b.shape == Shape::Cone) {
        ConeSpheres cs = MakeConeSpheres(b);
        for (int i = 0; i < 3; ++i) {
            float m = 0;
            for (int k = 0; k < 3; ++k) m = std::max(m, std::fabs(Comp(b.ax[1], i) * cs.c[k]) + cs.s[k]);
            SetComp(e, i, m);
        }
        return e;
    }
    for (int i = 0; i < 3; ++i)
        SetComp(e, i, b.half.x * std::fabs(Comp(b.ax[0], i)) + b.half.y * std::fabs(Comp(b.ax[1], i)) +
                          b.half.z * std::fabs(Comp(b.ax[2], i)));
    return e;
}

// Farthest point of the box in direction dir.
Vec3 Support(const Body& b, Vec3 dir) {
    auto s = [](float v) { return v >= 0 ? 1.0f : -1.0f; };
    return b.pos + b.ax[0] * (s(Dot(b.ax[0], dir)) * b.half.x) + b.ax[1] * (s(Dot(b.ax[1], dir)) * b.half.y) +
           b.ax[2] * (s(Dot(b.ax[2], dir)) * b.half.z);
}

// One contact: normal (a -> b), a single point, penetration depth.
struct Contact {
    Vec3 n, p;
    float depth = 0;
};

// OBB vs OBB: separating axis theorem over the 3 + 3 face axes and the 9 edge-edge axes.
// The axis with the smallest overlap is the contact normal; the contact point is midway between the two
// deepest corners.
bool ObbObb(const Body& a, const Body& b, Contact& c) {
    const Vec3 d = b.pos - a.pos;
    float best = 1e30f;
    bool separated = false;

    auto test = [&](Vec3 L) {
        float len = Length(L);
        if (len < 1e-4f) return;               // parallel edges: no axis
        L = L * (1.0f / len);
        float ra = 0, rb = 0;
        for (int k = 0; k < 3; ++k) {
            ra += Comp(a.half, k) * std::fabs(Dot(a.ax[k], L));
            rb += Comp(b.half, k) * std::fabs(Dot(b.ax[k], L));
        }
        float dist = Dot(d, L);
        float pen = ra + rb - std::fabs(dist);
        if (pen <= 0) { separated = true; return; }
        if (pen < best) {
            best = pen;
            c.n = dist >= 0 ? L : -L;
        }
    };

    for (int i = 0; i < 3; ++i) test(a.ax[i]);
    for (int i = 0; i < 3; ++i) test(b.ax[i]);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) test(Cross(a.ax[i], b.ax[j]));
    if (separated) return false;

    c.depth = best;
    c.p = (Support(a, c.n) + Support(b, -c.n)) * 0.5f;
    return true;
}

// a = sphere, b = oriented box. Closest point on the box to the sphere centre, found in the box's own axes.
bool SphereObb(const Body& a, const Body& b, Contact& c) {
    const float r = a.half.x;
    Vec3 rel = a.pos - b.pos;
    Vec3 closest = b.pos;
    float l[3];
    for (int k = 0; k < 3; ++k) {
        l[k] = Dot(rel, b.ax[k]);
        closest += b.ax[k] * Clampf(l[k], -Comp(b.half, k), Comp(b.half, k));
    }
    Vec3 diff = closest - a.pos;
    float d2 = Dot(diff, diff);
    if (d2 > r * r) return false;
    float dist = std::sqrt(d2);
    c.p = closest;
    if (dist > 1e-5f) {
        c.n = diff * (1.0f / dist);
        c.depth = r - dist;
    } else {   // centre inside the box: push out through the nearest face
        int axis = 0;
        float bestPen = 1e30f;
        for (int k = 0; k < 3; ++k) {
            float pen = Comp(b.half, k) - std::fabs(l[k]) + r;
            if (pen < bestPen) { bestPen = pen; axis = k; }
        }
        c.n = b.ax[axis] * (l[axis] >= 0 ? -1.0f : 1.0f);
        c.depth = bestPen;
    }
    return true;
}

bool Narrow(const Body& a, const Body& b, Contact& c) {
    bool sa = a.shape == Shape::Sphere, sb = b.shape == Shape::Sphere;
    if (sa && sb) {
        float depth;
        if (!SphereSphere(a, b, c.n, depth)) return false;
        c.depth = depth;
        c.p = a.pos + c.n * (a.half.x - depth * 0.5f);
        return true;
    }
    if (!sa && !sb) return ObbObb(a, b, c);
    if (sa) return SphereObb(a, b, c);
    if (!SphereObb(b, a, c)) return false;
    c.n = -c.n;
    return true;
}

// The collision parts of a body: a cone gives its three spheres (written into tmp), anything else is a
// single part, the body itself. Returns the number of parts.
int Parts(const Body& body, Body tmp[3], const Body* out[3]) {
    if (body.shape != Shape::Cone) {
        out[0] = &body;
        return 1;
    }
    ConeSpheres cs = MakeConeSpheres(body);
    for (int k = 0; k < 3; ++k) {
        tmp[k].shape = Shape::Sphere;
        tmp[k].pos = body.pos + body.ax[1] * cs.c[k];
        tmp[k].half = {cs.s[k], cs.s[k], cs.s[k]};
        out[k] = &tmp[k];
    }
    return 3;
}

// Impulse at the contact point p: linear response plus torque (r x J), with Coulomb friction.
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
        apply(t * std::max(-tl / effMass(t), -kFriction * jn));
    }
}
}  // namespace

void ObbWorld::InitBody(Body& b) { b.w = {Rand(-3, 3), Rand(-3, 3), Rand(-3, 3)}; }

void ObbWorld::OnKick(Body& b) { b.w += Vec3(Rand(-8, 8), Rand(-8, 8), Rand(-8, 8)); }

void ObbWorld::OnAdopt(Body& b) { b.w = {Rand(-3, 3), Rand(-3, 3), Rand(-3, 3)}; }   // visible spin right away

void ObbWorld::UpdateBody(Body& b, float h) {
    b.w = b.w * std::max(0.0f, 1.0f - kAngularDamping * h);
    Rotate(b, h);
    BounceOffWalls(b, WorldExtent(b));   // walls act on the box's axis-aligned extent: no tipping from walls
}

void ObbWorld::Resolve(Body& a, Body& b) {
    // cheap reject: bounding spheres (radius = half diagonal) must overlap first
    Vec3 d = b.pos - a.pos;
    float r = Length(a.half) + Length(b.half);
    if (Dot(d, d) >= r * r) return;

    // test every part of a against every part of b and keep the deepest contact
    Body tmpA[3], tmpB[3];
    const Body *partsA[3], *partsB[3];
    int na = Parts(a, tmpA, partsA), nb = Parts(b, tmpB, partsB);
    Contact c;
    bool hit = false;
    for (int i = 0; i < na; ++i)
        for (int j = 0; j < nb; ++j) {
            Contact t;
            if (Narrow(*partsA[i], *partsB[j], t) && (!hit || t.depth > c.depth)) {
                c = t;
                hit = true;
            }
        }
    if (!hit) return;
    SeparateBodies(a, b, c.n, c.depth);
    ContactImpulse(a, b, c.n, c.p);
}
