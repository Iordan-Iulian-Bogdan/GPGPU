#pragma once
#include <random>
#include <vector>
#include "instance.h"
#include "math3d.h"

// Prototip 1 - fizica pe CPU.
enum class Shape { Sphere = 0, Cube = 1, Cylinder = 2, Cone = 3 };

struct Body {
    Vec3 pos, vel;
    Vec3 half;          // Sphere: half.x = radius. Others: half extents of the bounding box (AABB).
    Vec3 color;
    Shape shape;
    float invMass;
    // Rotation (used only in OBB mode; in AABB mode the axes stay identity and w stays 0).
    Vec3 ax[3] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };   // orientation: local axes in world space
    Vec3 w;                                             // angular velocity (world space)
    Vec3 invI;                                          // inverse inertia, diagonal, in local axes
};

class World {
public:
    // Interior of the box: x in [-W,W], z in [-D,D], y in [0,H]. The face at z=-D is logically
    // present (objects collide with it) but not drawn, so the user can see inside.
    static constexpr float W = 7.0f, D = 7.0f, H = 20.0f;
    static constexpr size_t MaxBodies = 8192;

    World();
    void Clear();
    void Spawn(Shape s, bool atTop);      // random size, random velocity
    void LoadPreset(int index);           // 0..2 : the three predefined modes
    void Kick();                          // gives random motion to every body (wakes the scene)
    void Step(float dt);                  // advance simulation (CPU)
    void BuildInstances(std::vector<Instance> out[4]) const;  // [sphere, cube, cylinder, cone]
    const std::vector<Body>& Bodies() const { return bodies_; }
    void SetParallel(bool on) { parallel_ = on; }
    bool Parallel() const { return parallel_; }
    // Collision mode: false = AABB (axis-aligned, no rotation), true = OBB (oriented boxes, bodies rotate).
    void SetObb(bool on);
    bool Obb() const { return obb_; }

private:
    void CollideWallsObb(Body& b);
    void ResolveObb(Body& a, Body& b);
    void AddBody(Shape s, Vec3 pos, Vec3 vel);
    void SubStep(float h);
    void CollideWalls(Body& b);
    void Resolve(Body& a, Body& b);

    std::vector<Body> bodies_;
    // uniform grid for the broad phase
    std::vector<int> head_, next_;    std::mt19937 rng_;
    bool parallel_ = true;   // OpenMP on/off, switchable at runtime
    bool obb_ = false;       // AABB / OBB collision mode, switchable at runtime
    float Rand(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng_); }
};
