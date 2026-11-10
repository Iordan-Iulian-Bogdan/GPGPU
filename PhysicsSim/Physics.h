#pragma once
#include <random>
#include <vector>
#include "instance.h"
#include "math3d.h"

// Prototip 1 - fizica pe CPU.
enum class Shape { Sphere = 0, Cube = 1, Cylinder = 2, Cone = 3 };

struct Body {
    Vec3 pos, vel;
    Vec3 half;          // Sphere: half.x = radius. Others: half extents of the bounding box.
    Vec3 color;
    Shape shape;
    float invMass;
    // Rotation (used only by ObbWorld; AabbWorld keeps the axes at identity and w at 0).
    Vec3 ax[3] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };   // orientation: local axes in world space
    Vec3 w;                                             // angular velocity (world space)
    Vec3 invI;                                          // inverse inertia, diagonal, in local axes
};

// Common part of the simulation: bodies, spawning/presets, the substep loop, the uniform-grid broad phase
// (parallelised with OpenMP) and instance building. The collision model is supplied by the subclasses:
//   AabbWorld - axis-aligned boxes, no rotation        (AabbWorld.h)
//   ObbWorld  - oriented boxes, bodies rotate          (ObbWorld.h)
class World {
public:
    // Interior of the box: x in [-W,W], z in [-D,D], y in [0,H]. The face at z=-D is logically
    // present (objects collide with it) but not drawn, so the user can see inside.
    static constexpr float W = 7.0f, D = 7.0f, H = 20.0f;
    static constexpr size_t MaxBodies = 8192;

    World();
    virtual ~World() = default;

    void Clear();
    void Spawn(Shape s, bool atTop);      // random size, random velocity
    void LoadPreset(int index);           // 0..2 : the three predefined modes
    void Kick();                          // gives random motion to every body (wakes the scene)
    void Step(float dt);                  // advance simulation (CPU)
    void BuildInstances(std::vector<Instance> out[4]) const;  // [sphere, cube, cylinder, cone]
    const std::vector<Body>& Bodies() const { return bodies_; }
    void SetParallel(bool on) { parallel_ = on; }
    bool Parallel() const { return parallel_; }

    virtual bool Obb() const = 0;         // which collision model this world uses

    // Takes over the bodies and settings of another world (used when switching collision model at runtime).
    void AdoptState(const World& other);

protected:
    // ---- the parts a collision model must provide ----
    // Rotation (if any) and collisions with the six walls, after gravity and linear integration.
    virtual void UpdateBody(Body& b, float h) = 0;
    // Narrow phase + response for one pair of bodies.
    virtual void Resolve(Body& a, Body& b) = 0;
    // Optional hooks.
    virtual void InitBody(Body&) {}       // a new body was created
    virtual void OnKick(Body&) {}         // a body was kicked
    virtual void OnAdopt(Body&) {}        // a body was taken over from another world

    // Shared contact helper: pushes two overlapping bodies apart along n (a -> b), weighted by inverse mass.
    static void SeparateBodies(Body& a, Body& b, Vec3 n, float depth);

    float Rand(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng_); }

private:
    void AddBody(Shape s, Vec3 pos, Vec3 vel);
    void SubStep(float h);

    std::vector<Body> bodies_;
    // uniform grid for the broad phase
    std::vector<int> head_, next_;
    std::mt19937 rng_;
    bool parallel_ = true;   // OpenMP on/off, switchable at runtime
};
