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

// ---- substep part 1a: gravity, integration, rotation and walls, one thread per body ----
__global__ void IntegrateKernel(Body* bodies, int n, float h, int obbMode) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Body b = bodies[i];
    b.vel.y -= kGravity * h;
    b.pos += b.vel * h;
    if (obbMode) ObbUpdateBody(b, h);
    else AabbUpdateBody(b);
    bodies[i] = b;
}

// ---- substep part 1b: uniform grid (one work-group) ----
__global__ __launch_bounds__(kGridBlock) void GridKernel(Body* bodies, int n, Grid g, int* cellCount, int* cellStart,
                                                         int* cellCursor, int* cellOfBody, int* sorted) {
    __shared__ int scan[kGridBlock];
    const int tid = threadIdx.x;
    const int cells = g.cells();

    for (int c = tid; c < cells; c += kGridBlock) cellCount[c] = 0;
    __syncthreads();

    // counting sort by cell
    for (int i = tid; i < n; i += kGridBlock) {
        int cx, cy, cz;
        CellCoords(g, bodies[i].pos, cx, cy, cz);
        const int c = (cz * g.ny + cy) * g.nx + cx;
        cellOfBody[i] = c;
        atomicAdd(&cellCount[c], 1);
    }
    __syncthreads();

    // exclusive prefix sum of the cell counts: every thread owns a run of cells
    {
        const int per = (cells + kGridBlock - 1) / kGridBlock;
        const int first = tid * per, last = min(first + per, cells);
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
    for (int c = tid; c < cells; c += kGridBlock) {
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

// Cheap read-only test whether two bodies can be in contact at all. The exact test is the resolve function's job.
__device__ inline bool CouldTouch(const Body& a, const Body& b, int obbMode) {
    const Vec3 d = b.pos - a.pos;
    if (obbMode) {   // bounding spheres
        const float r = Length(a.half) + Length(b.half);
        return Dot(d, d) < r * r;
    }
    return a.half.x + b.half.x > fabsf(d.x) && a.half.y + b.half.y > fabsf(d.y) && a.half.z + b.half.z > fabsf(d.z);
}

// ---- substep part 2: the pair solver for the cells of one colour; one wave (work-group) per cell ----
// For every body of the cell the 27 lanes first scan the 27 neighbour cells at the same time (read only) and note the
// bodies that could touch it; then the wave resolves those pairs one after another in the order of the cell loops
// (dz, dy, dx) of the CPU version. Scanning is where nearly all the time went: it is a long chain of dependent
// loads when done by one lane, but cheap when each lane does one cell. The scan sees the body as it was before its
// first pair of this sweep; a pair that only starts to overlap because of an earlier push in the same sweep is
// caught in the next substep.
constexpr int kCand = 32;   // candidates remembered per neighbour cell (a rare excess takes a slower path)

__global__ __launch_bounds__(obb::kWave) void PairsKernel(Body* bodies, int obbMode, Grid g, int color,
                                                          const int* cellCount, const int* cellStart,
                                                          const int* sorted) {
    // shared storage of the OBB part pairs (raw bytes, Box has a constructor)
    __shared__ char partStore[obb::WaveScratch::kBytes] __attribute__((aligned(16)));
    __shared__ int candCount[obb::kWave];
    __shared__ int cand[27][kCand];
    obb::WaveScratch scratch = obb::WaveScratch::Carve(partStore);

    const int lane = threadIdx.x;
    const int ox = color % 3, oy = (color / 3) % 3, oz = color / 9;
    const int cxn = (g.nx - ox + 2) / 3, cyn = (g.ny - oy + 2) / 3;
    const int idx = blockIdx.x;
    const int cx = ox + (idx % cxn) * 3;
    const int cy = oy + ((idx / cxn) % cyn) * 3;
    const int cz = oz + (idx / (cxn * cyn)) * 3;
    const int own = (cz * g.ny + cy) * g.nx + cx;

    // neighbour cell of a lane (lanes 0..26 map to dx, dy, dz in the same order as the CPU loops), -1 = outside
    auto neighbour = [&](int l) {
        const int x = cx + l % 3 - 1, y = cy + l / 3 % 3 - 1, z = cz + l / 9 - 1;
        return (x < 0 || y < 0 || z < 0 || x >= g.nx || y >= g.ny || z >= g.nz) ? -1 : (z * g.ny + y) * g.nx + x;
    };
    auto resolve = [&](Body& a, Body& b) {
        // every lane resolves the same pair (identical writes); the OBB test shares the part pairs out
        if (obbMode) obb::ObbResolveWave(a, b, scratch);
        else AabbResolve(a, b);
    };
    const int myNeighbour = lane < 27 ? neighbour(lane) : -1;

    for (int ia = 0; ia < cellCount[own]; ++ia) {
        const int i = sorted[cellStart[own] + ia];
        Body mine = bodies[i];   // this wave owns the whole neighbourhood during the colour, so a copy is safe
        if (obbMode) obb::PrepareA(mine, scratch);   // its parts, once for all the pairs of this sweep

        // 1. gather the candidates, one neighbour cell per lane
        int found = 0;
        if (myNeighbour >= 0) {
            const int* list = sorted + cellStart[myNeighbour];
            const int cnt = cellCount[myNeighbour];
            for (int ja = 0; ja < cnt; ++ja) {
                const int j = list[ja];
                if (j <= i || !CouldTouch(mine, bodies[j], obbMode)) continue;
                if (found < kCand) cand[lane][found] = j;
                ++found;
            }
        }
        candCount[lane] = found;
        __syncthreads();

        // 2. resolve them in lane order
        for (int l = 0; l < 27; ++l) {
            const int total = candCount[l];
            for (int k = 0; k < total && k < kCand; ++k) resolve(mine, bodies[cand[l][k]]);
            if (total > kCand) {   // rare: walk the cell again, past the candidates already handled
                const int* list = sorted + cellStart[neighbour(l)];
                const int cnt = cellCount[neighbour(l)];
                int seen = 0;
                for (int ja = 0; ja < cnt; ++ja) {
                    const int j = list[ja];
                    if (j <= i || !CouldTouch(mine, bodies[j], obbMode)) continue;
                    if (seen++ >= kCand) resolve(mine, bodies[j]);
                }
            }
        }
        bodies[i] = mine;
        __syncthreads();   // before the next body overwrites cand / candCount
    }
}
// ---- queueing one substep (asynchronous) ----
void LaunchGrid(Sim* s, float h, int obbMode) {
    IntegrateKernel<<<(s->count + 63) / 64, 64>>>(s->bodies, s->count, h, obbMode);
    GridKernel<<<1, kGridBlock>>>(s->bodies, s->count, GridFor(obbMode != 0), s->cellCount, s->cellStart,
                                  s->cellCursor, s->cellOfBody, s->sorted);
}

void LaunchPairs(Sim* s, int obbMode) {
    const Grid g = GridFor(obbMode != 0);
    for (int color = 0; color < kColors; ++color) {
        const int ox = color % 3, oy = color / 3 % 3, oz = color / 9;
        const int cells = ((g.nx - ox + 2) / 3) * ((g.ny - oy + 2) / 3) * ((g.nz - oz + 2) / 3);
        PairsKernel<<<cells, obb::kWave>>>(s->bodies, obbMode, g, color, s->cellCount, s->cellStart, s->sorted);
    }
}

void LaunchSubStep(Sim* s, float h, int obbMode) {
    LaunchGrid(s, h, obbMode);
    LaunchPairs(s, obbMode);
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
              Check(hipMalloc(&s->cellCount, sizeof(int) * kMaxCells), "hipMalloc cellCount") &&
              Check(hipMalloc(&s->cellStart, sizeof(int) * kMaxCells), "hipMalloc cellStart") &&
              Check(hipMalloc(&s->cellCursor, sizeof(int) * kMaxCells), "hipMalloc cellCursor") &&
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
        LaunchSubStep(s, h, obbMode);
    }
    if (!Check(hipGetLastError(), "kernel launch") || !Check(hipEventRecord(s->stop), "hipEventRecord") ||
        !Check(hipEventSynchronize(s->stop), "kernel execution"))
        return 0;
    float ms = 0;
    if (!Check(hipEventElapsedTime(&ms, s->start, s->stop), "hipEventElapsedTime")) return 0;
    if (gpuMs) *gpuMs = ms;
    return 1;
}

// Profiling aid (not used by the app): like ph_step, but reports how long the grid kernels and the 27 pair
// launches took in total.
PH_EXPORT int ph_profile(void* handle, float h, int substeps, int obbMode, float* gridMs, float* pairsMs) {
    Sim* s = static_cast<Sim*>(handle);
    *gridMs = *pairsMs = 0;
    if (s->count == 0 || substeps < 1 || substeps > 4) return 0;
    hipEvent_t ev[9];
    for (auto& e : ev)
        if (!Check(hipEventCreate(&e), "hipEventCreate")) return 0;
    (void)hipEventRecord(ev[0]);
    for (int step = 0; step < substeps; ++step) {
        LaunchGrid(s, h, obbMode);
        (void)hipEventRecord(ev[2 * step + 1]);
        LaunchPairs(s, obbMode);
        (void)hipEventRecord(ev[2 * step + 2]);
    }
    bool ok = Check(hipEventSynchronize(ev[2 * substeps]), "profile");
    for (int step = 0; ok && step < substeps; ++step) {
        float g = 0, p = 0;
        (void)hipEventElapsedTime(&g, ev[2 * step], ev[2 * step + 1]);
        (void)hipEventElapsedTime(&p, ev[2 * step + 1], ev[2 * step + 2]);
        *gridMs += g;
        *pairsMs += p;
    }
    for (auto& e : ev) (void)hipEventDestroy(e);
    return ok;
}

PH_EXPORT int ph_download(void* handle, Body* bodies, int n) {
    Sim* s = static_cast<Sim*>(handle);
    return n == 0 || Check(hipMemcpy(bodies, s->bodies, sizeof(Body) * n, hipMemcpyDeviceToHost), "download");
}
}
