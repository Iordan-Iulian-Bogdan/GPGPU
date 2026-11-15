#pragma once
#include <memory>
#include <string>

#include "AabbWorld.h"
#include "ObbWorld.h"

// GPU backend (AMD HIP). The collision model (AABB or OBB) is still chosen by the base class; HipWorld only
// changes WHERE Step runs: the bodies live on the GPU and one kernel launch per frame advances all substeps,
// using the same collision code as the CPU worlds (PhysicsCommon.h / AabbCollision.h / ObbCollision.h).
// The GPU code is in PhysicsHip.dll, which is loaded at run time: without the DLL, a HIP runtime or an AMD GPU the
// factory below just returns null and the app keeps using the CPU worlds.

// Loads PhysicsHip.dll (next to the exe) once and creates the device state. On success returns true and sets
// `info` to the GPU name, otherwise `info` says why not. The result is cached.
bool HipAvailable(std::string& info);

template <class Base>
class HipWorld : public Base {
public:
    HipWorld();
    ~HipWorld() override;
    HipWorld(const HipWorld&) = delete;
    HipWorld& operator=(const HipWorld&) = delete;

    bool Ok() const { return handle_ != nullptr; }   // false if the device state could not be created
    bool Gpu() const override { return handle_ != nullptr; }
    float GpuMs() const override { return gpuMs_; }
    void Step(float dt) override;

private:
    void Fail(const char* what);   // reports the error and falls back to the CPU for good

    void* handle_ = nullptr;
    float gpuMs_ = 0;
};

using HipAabbWorld = HipWorld<AabbWorld>;
using HipObbWorld = HipWorld<ObbWorld>;

// A GPU world with the given collision model, or null (and `info` filled in) if the GPU backend is unavailable.
std::unique_ptr<World> MakeHipWorld(bool obb, std::string& info);
