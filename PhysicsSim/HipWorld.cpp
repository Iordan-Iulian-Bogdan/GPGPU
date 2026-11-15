#include "HipWorld.h"

#include <windows.h>

#include <cstdio>

#include "PhysicsCommon.h"

namespace {
// The C interface of PhysicsHip.dll (hip/PhysicsHip.hip).
struct HipApi {
    HMODULE dll = nullptr;
    const char* (*error)() = nullptr;
    int (*bodySize)() = nullptr;
    void* (*create)(int capacity, char* name, int nameLen) = nullptr;
    void (*destroy)(void*) = nullptr;
    int (*upload)(void*, const Body*, int) = nullptr;
    int (*step)(void*, float, int, int, float*) = nullptr;
    int (*download)(void*, Body*, int) = nullptr;
};

HipApi g_api;
bool g_tried = false, g_ok = false;
std::string g_info;

template <class F>
bool Bind(F& fn, const char* name) {
    fn = reinterpret_cast<F>(GetProcAddress(g_api.dll, name));
    return fn != nullptr;
}

bool LoadApi() {
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring path = exe;
    path = path.substr(0, path.find_last_of(L"\\/") + 1) + L"PhysicsHip.dll";

    if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES) {
        g_info = "PhysicsHip.dll not found next to the exe (it is built by hip\\build_hip.bat and needs the AMD HIP SDK)";
        return false;
    }
    g_api.dll = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g_api.dll) {
        g_info = "PhysicsHip.dll could not be loaded (error " + std::to_string(GetLastError()) +
                 "): the AMD HIP runtime (amdhip64_7.dll) is missing; install a recent AMD Adrenalin driver or the HIP SDK";
        return false;
    }
    if (!(Bind(g_api.error, "ph_error") && Bind(g_api.bodySize, "ph_body_size") && Bind(g_api.create, "ph_create") &&
          Bind(g_api.destroy, "ph_destroy") && Bind(g_api.upload, "ph_upload") && Bind(g_api.step, "ph_step") &&
          Bind(g_api.download, "ph_download"))) {
        g_info = "PhysicsHip.dll does not have the expected functions";
        return false;
    }
    if (g_api.bodySize() != static_cast<int>(sizeof(Body))) {
        g_info = "PhysicsHip.dll was built for a different Body layout: rebuild it";
        return false;
    }
    return true;
}
}  // namespace

bool HipAvailable(std::string& info) {
    if (!g_tried) {
        g_tried = true;
        if (!LoadApi()) {
            g_ok = false;
        } else {
            // probe: create and drop a device state to find out whether there is a usable GPU
            char name[256] = "";
            void* probe = g_api.create(static_cast<int>(World::MaxBodies), name, sizeof name);
            if (probe) {
                g_api.destroy(probe);
                g_info = name;
                g_ok = true;
            } else {
                g_info = std::string("HIP is not usable: ") + g_api.error();
                g_ok = false;
            }
        }
    }
    info = g_info;
    return g_ok;
}

template <class Base>
HipWorld<Base>::HipWorld() {
    std::string info;
    if (HipAvailable(info)) handle_ = g_api.create(static_cast<int>(World::MaxBodies), nullptr, 0);
}

template <class Base>
HipWorld<Base>::~HipWorld() {
    if (handle_) g_api.destroy(handle_);
}

template <class Base>
void HipWorld<Base>::Fail(const char* what) {
    std::printf("GPU backend failed (%s: %s): continuing on the CPU\n", what, g_api.error());
    g_api.destroy(handle_);
    handle_ = nullptr;
    gpuMs_ = 0;
}

template <class Base>
void HipWorld<Base>::Step(float dt) {
    if (!handle_) {   // no GPU: behave like the CPU world we derive from
        Base::Step(dt);
        return;
    }
    const int n = static_cast<int>(this->bodies_.size());
    if (this->bodiesChanged_) {   // spawn / clear / kick / mode switch happened on the host
        if (!g_api.upload(handle_, this->bodies_.data(), n)) return Fail("upload");
        this->bodiesChanged_ = false;
    }
    if (n == 0) return;
    if (!g_api.step(handle_, dt / phys::kSubSteps, phys::kSubSteps, this->Obb() ? 1 : 0, &gpuMs_)) return Fail("step");
    // The host copy is what BuildInstances draws and what a switch to another world adopts.
    if (!g_api.download(handle_, this->bodies_.data(), n)) return Fail("download");
}

template class HipWorld<AabbWorld>;
template class HipWorld<ObbWorld>;

std::unique_ptr<World> MakeHipWorld(bool obb, std::string& info) {
    if (!HipAvailable(info)) return nullptr;
    std::unique_ptr<World> world;
    bool ok;
    if (obb) {
        auto w = std::make_unique<HipObbWorld>();
        ok = w->Ok();
        world = std::move(w);
    } else {
        auto w = std::make_unique<HipAabbWorld>();
        ok = w->Ok();
        world = std::move(w);
    }
    if (!ok) {
        info = "could not create the GPU state";
        return nullptr;
    }
    return world;
}
