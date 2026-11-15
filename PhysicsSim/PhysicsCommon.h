#pragma once
// Constants and small helpers shared by the collision models. Everything here is PHYS_HD, so the very same
// code runs in the CPU worlds and in the HIP kernels (hip/PhysicsHip.hip).
#include <algorithm>
#include <cmath>
#include "PhysicsTypes.h"

namespace phys {
inline constexpr float kGravity = 9.81f;
inline constexpr float kRestitution = 0.35f;
inline constexpr float kFriction = 0.4f;
inline constexpr float kSlop = 0.005f;
inline constexpr float kCorrection = 0.8f;
inline constexpr float kPi = 3.14159265f;
inline constexpr int kSubSteps = 2;

// Uniform grid of the broad phase. A body only has to be tested against the bodies in the 3x3x3 cells around its
// own, which is enough if a cell is at least as wide as the largest distance (on one axis) at which two bodies can
// still touch. That depends on the collision model, so each model has its own grid:
//  - AABB: two boxes touch when their centres are closer than the sum of the half extents on every axis, and no
//          half extent is larger than 0.4 (sphere radius), so 0.8 on one axis.
//  - OBB:  two bounding spheres of radius 0.61 (a rotated cube's half-diagonal) = 1.22.
// Smaller cells hold fewer bodies, so the AABB grid has far fewer pair candidates.
struct Grid {
    float cell;
    int nx, ny, nz;
    PHYS_HD constexpr int cells() const { return nx * ny * nz; }
};
PHYS_HD constexpr Grid MakeGrid(float cell) {
    return {cell, (int)(2 * kBoxW / cell) + 1, (int)(kBoxH / cell) + 1, (int)(2 * kBoxD / cell) + 1};
}
PHYS_HD constexpr Grid GridAabb() { return MakeGrid(0.85f); }
PHYS_HD constexpr Grid GridObb() { return MakeGrid(1.25f); }
PHYS_HD constexpr Grid GridFor(bool obb) { return obb ? GridObb() : GridAabb(); }
// Room for the cell arrays of either grid.
inline constexpr int kMaxCells = GridAabb().cells() > GridObb().cells() ? GridAabb().cells() : GridObb().cells();

PHYS_HD inline float Clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
PHYS_HD inline int Clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
PHYS_HD inline float Comp(const Vec3& v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
PHYS_HD inline void SetComp(Vec3& v, int i, float f) { (i == 0 ? v.x : (i == 1 ? v.y : v.z)) = f; }

// Grid cell of a position.
PHYS_HD inline void CellCoords(const Grid& g, const Vec3& p, int& cx, int& cy, int& cz) {
    cx = Clampi((int)((p.x + kBoxW) / g.cell), 0, g.nx - 1);
    cy = Clampi((int)(p.y / g.cell), 0, g.ny - 1);
    cz = Clampi((int)((p.z + kBoxD) / g.cell), 0, g.nz - 1);
}

// Contact between a and b. n points from a to b.
PHYS_HD inline bool SphereSphere(const Body& a, const Body& b, Vec3& n, float& depth) {
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
PHYS_HD inline void BounceOffWalls(Body& b, Vec3 half) {
    const float lo[3] = {-kBoxW, 0, -kBoxD}, hi[3] = {kBoxW, kBoxH, kBoxD};
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

// Shared contact helper: pushes two overlapping bodies apart along n (a -> b), weighted by inverse mass.
PHYS_HD inline void SeparateBodies(Body& a, Body& b, Vec3 n, float depth) {
    float sum = a.invMass + b.invMass;
    float corr = std::max(depth - kSlop, 0.0f) * kCorrection / sum;
    corr = std::min(corr, 0.5f);
    a.pos -= n * (corr * a.invMass);
    b.pos += n * (corr * b.invMass);
}
}  // namespace phys
