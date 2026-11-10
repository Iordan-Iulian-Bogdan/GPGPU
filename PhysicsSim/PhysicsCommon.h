#pragma once
// Constants and small helpers shared by the collision models (internal to the physics code).
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
}  // namespace phys
