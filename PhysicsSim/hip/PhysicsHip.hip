// AMD HIP backend of the physics. Built as PhysicsHip.dll (see build_hip.bat) and loaded at run time by the
// app, so everything still works on a PC without a HIP SDK / AMD GPU.
//
// The GPU runs the same algorithm as the CPU worlds, with the same collision code (PhysicsCommon.h,
// AabbCollision.h and ObbCollision.h are compiled for both). One frame = kSubSteps substeps, and each substep is
//   1. GridKernel      gravity, integration, rotation, walls (one thread per body), then the uniform grid by
//                      counting sort (one work-group); each cell's list is ordered by body index
//   2. PairsKernel x27 the pair solver. Like the OpenMP version it is a sequential-impulse solver, so parallelism
//                      comes from grid colouring: the cells of one colour are >= 3 cells apart and never touch the
//                      same body, so a launch handles all cells of one colour at once, and the 27 colours follow
//                      one another in the stream. One WAVE per cell: it walks the cell's body pairs in lockstep,
//                      and for a pair of cones / cylinders the lanes split the (up to 64 x 64) part pairs.
// The launches are queued without waiting; the host synchronises once per frame.
#include <hip/hip_runtime.h>

#include <cstdio>

#include "../AabbCollision.h"
#include "../ObbCollision.h"
#include "../ObbWave.h"
#include "../PhysicsCommon.h"

using namespace phys;

// The device pass of the compiler ignores dllexport (with a warning), the host pass honours it.
#ifdef __HIP_DEVICE_COMPILE__
#define PH_EXPORT
#else
#define PH_EXPORT __declspec(dllexport)
#endif

namespace {
constexpr int kGridBlock = 256;   // threads of the grid work-group
constexpr int kColors = 27;

char g_error[256] = "";

bool Check(hipError_t e, const char* what) {
    if (e == hipSuccess) return true;
    std::snprintf(g_error, sizeof g_error, "%s: %s", what, hipGetErrorString(e));
    return false;
}

// Frees device memory; failures while cleaning up are not actionable.
template <class T> void Free(T*& p) {
    if (p) (void)hipFree(p);
    p = nullptr;
}

struct Sim {
    Body* bodies = nullptr;
    int* cellCount = nullptr;    // bodies per cell
    int* cellStart = nullptr;    // first slot of each cell in `sorted`
    int* cellCursor = nullptr;   // fill counter while scattering
    int* cellOfBody = nullptr;
    int* sorted = nullptr;       // body indices grouped by cell
    int capacity = 0, count = 0;
    hipEvent_t start = nullptr, stop = nullptr;
    char name[256] = "";
};

// ---- substep part 1: integration and uniform grid (one work-group) ----
__global__ __launch_bounds__(kGridBlock) void GridKernel(Body* bodies, int n, float h, int obbMode, int* cellCount,
                                                         int* cellStart, int* cellCursor, int* cellOfBody,
                                                         int* sorted) {
    __shared__ int scan[kGridBlock];
    const int tid = threadIdx.x;

    // gravity + integration + walls
    for (int i = tid; i < n; i += kGridBlock) {
        Body b = bodies[i];
        b.vel.y -= kGravity * h;
        b.pos += b.vel * h;
        if (obbMode) ObbUpdateBody(b, h);
        else AabbUpdateBody(b);
        bodies[i] = b;
    }
    for (int c = tid; c < kCells; c += kGridBlock) cellCount[c] = 0;
    __syncthreads();

    // counting sort by cell
    for (int i = tid; i < n; i += kGridBlock) {
        int cx, cy, cz;
        CellCoords(bodies[i].pos, cx, cy, cz);
        const int c = (cz * NY + cy) * NX + cx;
        cellOfBody[i] = c;
        atomicAdd(&cellCount[c], 1);
    }
    __syncthreads();

    // exclusive prefix sum of the cell counts: every thread owns a run of cells
    {
        const int per = (kCells + kGridBlock - 1) / kGridBlock;
        const int first = tid * per, last = min(first + per, kCells);
        int mine = 0;
        for (int c = first; c < last; ++c) mine += cellCount[c];
        scan[tid] = mine;
        __syncthreads();
        for (int off = 1; off < kGridBlock; off <<= 1) {   // Hillis-Steele inclusive scan
            const int v = tid >= off ? scan[tid - off] : 0;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        int run = scan[tid] - mine;
        for (int c = first; c < last; ++c) {
            cellStart[c] = run;
            cellCursor[c] = 0;
            run += cellCount[c];
        }
    }
    __syncthreads();

    for (int i = tid; i < n; i += kGridBlock) {
        const int c = cellOfBody[i];
        sorted[cellStart[c] + atomicAdd(&cellCursor[c], 1)] = i;
    }
    __syncthreads();

    // The scatter order depends on thread timing. Sorting every cell's list into descending body index makes the
    // solver repeatable from run to run (it is the order of the CPU's linked lists).
    for (int c = tid; c < kCells; c += kGridBlock) {
        int* list = sorted + cellStart[c];
        const int cnt = cellCount[c];
        for (int a = 1; a < cnt; ++a) {
            const int v = list[a];
            int k = a - 1;
            while (k >= 0 && list[k] < v) {
                list[k + 1] = list[k];
                --k;
            }
            list[k + 1] = v;
        }
    }
}

// ---- substep part 2: the pair solver for the cells of one colour; one wave (work-group) per cell ----
__global__ __launch_bounds__(obb::kWave) void PairsKernel(Body* bodies, int obbMode, int color, const int* cellCount,
                                                          const int* cellStart, const int* sorted) {
    // shared storage for the parts of the two bodies being resolved (Box has a constructor, so raw bytes)
    __shared__ char partStore[2 * obb::kMaxParts * sizeof(obb::Box)] __attribute__((aligned(16)));
    obb::Box* sa = reinterpret_cast<obb::Box*>(partStore);
    obb::Box* sb = sa + obb::kMaxParts;

    const int ox = color % 3, oy = (color / 3) % 3, oz = color / 9;
    const int cxn = (NX - ox + 2) / 3, cyn = (NY - oy + 2) / 3;
    const int idx = blockIdx.x;
    const int cx = ox + (idx % cxn) * 3;
    const int cy = oy + ((idx / cxn) % cyn) * 3;
    const int cz = oz + (idx / (cxn * cyn)) * 3;
    const int own = (cz * NY + cy) * NX + cx;

    for (int ia = 0; ia < cellCount[own]; ++ia) {
        const int i = sorted[cellStart[own] + ia];
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int x = cx + dx, y = cy + dy, z = cz + dz;
                    if (x < 0 || y < 0 || z < 0 || x >= NX || y >= NY || z >= NZ) continue;
                    const int nb = (z * NY + y) * NX + x;
                    const int* list = sorted + cellStart[nb];
                    const int cnt = cellCount[nb];
                    for (int ja = 0; ja < cnt; ++ja) {
                        const int j = list[ja];
                        if (j <= i) continue;
                        // All lanes of the wave follow the same path. For the cheap tests every lane does the same
                        // work on the same bodies (identical writes); the OBB test shares the part pairs out.
                        if (obbMode) obb::ObbResolveWave(bodies[i], bodies[j], sa, sb);
                        else AabbResolve(bodies[i], bodies[j]);
                    }
                }
    }
}
}  // namespace

extern "C" {
// All functions return 0 on failure (see ph_error), except where noted.

PH_EXPORT const char* ph_error() { return g_error; }

// Size of Body as the DLL sees it; the app refuses to use the DLL if it differs from its own.
PH_EXPORT int ph_body_size() { return (int)sizeof(Body); }

// Creates the device state; the device name is copied into nameOut. Returns null on failure.
PH_EXPORT void* ph_create(int capacity, char* nameOut, int nameLen) {
    int devices = 0;
    if (!Check(hipGetDeviceCount(&devices), "hipGetDeviceCount")) return nullptr;
    if (devices == 0) {
        std::snprintf(g_error, sizeof g_error, "no HIP device found");
        return nullptr;
    }
    Sim* s = new Sim;
    s->capacity = capacity;
    bool ok = Check(hipSetDevice(0), "hipSetDevice") &&
              Check(hipMalloc(&s->bodies, sizeof(Body) * capacity), "hipMalloc bodies") &&
              Check(hipMalloc(&s->cellCount, sizeof(int) * kCells), "hipMalloc cellCount") &&
              Check(hipMalloc(&s->cellStart, sizeof(int) * kCells), "hipMalloc cellStart") &&
              Check(hipMalloc(&s->cellCursor, sizeof(int) * kCells), "hipMalloc cellCursor") &&
              Check(hipMalloc(&s->cellOfBody, sizeof(int) * capacity), "hipMalloc cellOfBody") &&
              Check(hipMalloc(&s->sorted, sizeof(int) * capacity), "hipMalloc sorted") &&
              Check(hipEventCreate(&s->start), "hipEventCreate") && Check(hipEventCreate(&s->stop), "hipEventCreate") &&
              Check(hipDeviceGetName(s->name, sizeof s->name, 0), "hipDeviceGetName");
    if (!ok) {
        Free(s->bodies); Free(s->cellCount); Free(s->cellStart); Free(s->cellCursor);
        Free(s->cellOfBody); Free(s->sorted);
        delete s;
        return nullptr;
    }
    if (nameOut && nameLen > 0) std::snprintf(nameOut, nameLen, "%s", s->name);
    return s;
}

PH_EXPORT void ph_destroy(void* handle) {
    Sim* s = static_cast<Sim*>(handle);
    if (!s) return;
    Free(s->bodies); Free(s->cellCount); Free(s->cellStart); Free(s->cellCursor);
    Free(s->cellOfBody); Free(s->sorted);
    (void)hipEventDestroy(s->start);
    (void)hipEventDestroy(s->stop);
    delete s;
}

PH_EXPORT int ph_upload(void* handle, const Body* bodies, int n) {
    Sim* s = static_cast<Sim*>(handle);
    if (n > s->capacity) {
        std::snprintf(g_error, sizeof g_error, "too many bodies (%d > %d)", n, s->capacity);
        return 0;
    }
    s->count = n;
    return n == 0 || Check(hipMemcpy(s->bodies, bodies, sizeof(Body) * n, hipMemcpyHostToDevice), "upload");
}

// Runs one frame: `substeps` substeps of length h. Returns the GPU time in milliseconds through gpuMs.
PH_EXPORT int ph_step(void* handle, float h, int substeps, int obbMode, float* gpuMs) {
    Sim* s = static_cast<Sim*>(handle);
    if (gpuMs) *gpuMs = 0;
    if (s->count == 0) return 1;
    if (!Check(hipEventRecord(s->start), "hipEventRecord")) return 0;
    for (int step = 0; step < substeps; ++step) {
        GridKernel<<<1, kGridBlock>>>(s->bodies, s->count, h, obbMode, s->cellCount, s->cellStart, s->cellCursor,
                                      s->cellOfBody, s->sorted);
        for (int color = 0; color < kColors; ++color) {
            const int ox = color % 3, oy = color / 3 % 3, oz = color / 9;
            const int cells = ((NX - ox + 2) / 3) * ((NY - oy + 2) / 3) * ((NZ - oz + 2) / 3);
            PairsKernel<<<cells, obb::kWave>>>(s->bodies, obbMode, color, s->cellCount, s->cellStart, s->sorted);
        }
    }
    if (!Check(hipGetLastError(), "kernel launch") || !Check(hipEventRecord(s->stop), "hipEventRecord") ||
        !Check(hipEventSynchronize(s->stop), "kernel execution"))
        return 0;
    float ms = 0;
    if (!Check(hipEventElapsedTime(&ms, s->start, s->stop), "hipEventElapsedTime")) return 0;
    if (gpuMs) *gpuMs = ms;
    return 1;
}

PH_EXPORT int ph_download(void* handle, Body* bodies, int n) {
    Sim* s = static_cast<Sim*>(handle);
    return n == 0 || Check(hipMemcpy(bodies, s->bodies, sizeof(Body) * n, hipMemcpyDeviceToHost), "download");
}
}
