#pragma once
// Constants and small helpers shared by the collision models (internal to the physics code).
#include <algorithm>
#include <cmath>
#include "Physics.h"

namespace phys {
inline constexpr float kGravity = 9.81f;
inline constexpr float kRestitution = 0.35f;
inline constexpr float kFriction = 0.4f;
inline constexpr float kSlop = 0.005f;
inline constexpr float kCorrection = 0.8f;
inline constexpr float kPi = 3.14159265f;

inline float Clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline int Clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float Comp(const Vec3& v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
inline void SetComp(Vec3& v, int i, float f) { (i == 0 ? v.x : (i == 1 ? v.y : v.z)) = f; }

// Contact between a and b. n points from a to b.
inline bool SphereSphere(const Body& a, const Body& b, Vec3& n, float& depth) {
    Vec3 d = b.pos - a.pos;
    float r = a.half.x + b.half.x, d2 = Dot(d, d);
    if (d2 >= r * r) return false;
    float dist = std::sqrt(d2);
    n = dist > 1e-6f ? d * (1.0f / dist) : Vec3(0, 1, 0);
    depth = r - dist;
    return true;
}
// Keeps the body inside the box (axis-aligned walls); "half" is the body's extent along each world axis.
// Bounces with restitution and applies Coulomb friction on the tangential velocity.
inline void BounceOffWalls(Body& b, Vec3 half) {
    const float lo[3] = {-World::W, 0, -World::D}, hi[3] = {World::W, World::H, World::D};
    for (int i = 0; i < 3; ++i) {
        for (int side = 0; side < 2; ++side) {
            float h = Comp(half, i);
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
}  // namespace phys
