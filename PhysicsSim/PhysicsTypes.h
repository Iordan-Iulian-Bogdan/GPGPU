#pragma once
// Plain data shared by the CPU physics, the collision functions (PhysicsCommon.h, AabbCollision.h,
// ObbCollision.h) and the HIP (GPU) backend. Keep this header free of the STL so GPU code can include it.
#include "math3d.h"

enum class Shape { Sphere = 0, Cube = 1, Cylinder = 2, Cone = 3 };

struct Body {
    Vec3 pos, vel;
    Vec3 half;          // Sphere: half.x = radius. Others: half extents of the bounding box.
    Vec3 color;
    Shape shape;
    float invMass;
    // Rotation (used only by the OBB model; the AABB model keeps the axes at identity and w at 0).
    Vec3 ax[3] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };   // orientation: local axes in world space
    Vec3 w;                                             // angular velocity (world space)
    Vec3 invI;                                          // inverse inertia, diagonal, in local axes
};

// The CPU code (MSVC) and the GPU backend (clang) must agree on this layout byte for byte.
static_assert(sizeof(Body) == 116, "Body layout changed: rebuild PhysicsHip.dll and update the ABI check");

namespace phys {
// Interior of the box: x in [-W,W], z in [-D,D], y in [0,H].
inline constexpr float kBoxW = 7.0f, kBoxD = 7.0f, kBoxH = 20.0f;
}  // namespace phys
