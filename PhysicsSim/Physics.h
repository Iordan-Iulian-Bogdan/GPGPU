#pragma once
#include <random>
#include <vector>
#include "PhysicsTypes.h"
#include "instance.h"

// Common part of the simulation: bodies, spawning/presets, the substep loop, the uniform-grid broad phase
// (parallelised with OpenMP) and instance building. The collision model is supplied by the subclasses:
//   AabbWorld - axis-aligned boxes, no rotation        (AabbWorld.h)
//   ObbWorld  - oriented boxes, bodies rotate          (ObbWorld.h)
class World {
public:
    // Interior of the box: x in [-W,W], z in [-D,D], y in [0,H]. The face at z=-D is logically
    // present (objects collide with it) but not drawn, so the user can see inside.
    static constexpr float W = phys::kBoxW, D = phys::kBoxD, H = phys::kBoxH;
    static constexpr size_t MaxBodies = 8192;

    World();
    virtual ~World() = default;

    void Clear();
    void Spawn(Shape s, bool atTop);      // random size, random velocity
    void LoadPreset(int index);           // 0..2 : the three predefined modes
    void Kick();                          // gives random motion to every body (wakes the scene)
    virtual void Step(float dt);          // advance the simulation (CPU here; a backend may run it elsewhere)
    void BuildInstances(std::vector<Instance> out[4]) const;  // [sphere, cube, cylinder, cone]
    const std::vector<Body>& Bodies() const { return bodies_; }
    void SetParallel(bool on) { parallel_ = on; }
    bool Parallel() const { return parallel_; }

    virtual bool Obb() const = 0;         // does this world rotate its bodies (OBB-style)?
    virtual bool Gpu() const { return false; }      // does Step run on the GPU?
    virtual float GpuMs() const { return 0; }       // GPU time of the last Step (kernel only), 0 on the CPU

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

    float Rand(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng_); }

    std::vector<Body> bodies_;
    // Set whenever the bodies are changed on the host (spawn, clear, kick, adopt). A backend that keeps its own
    // copy of the bodies (the GPU) re-uploads when it is set and clears it afterwards.
    bool bodiesChanged_ = true;

private:
    void AddBody(Shape s, Vec3 pos, Vec3 vel);
    void SubStep(float h, bool reverse);

    // uniform grid for the broad phase
    std::vector<int> head_, next_;
    std::mt19937 rng_;
    bool parallel_ = true;   // OpenMP on/off, switchable at runtime
};
