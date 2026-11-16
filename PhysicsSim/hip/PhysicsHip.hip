// AMD HIP backend of the physics. Built as PhysicsHip.dll (see build_hip.bat) and loaded at run time by the
// app, so everything still works on a PC without a HIP SDK / AMD GPU.
//
// The GPU uses the same collision code as the CPU worlds (PhysicsCommon.h, AabbCollision.h and ObbCollision.h are
// compiled for both) and the same kind of solver, but splits the work so that the expensive part runs on the whole
// GPU at once. One frame = kSubSteps substeps, and each substep is
//   1. IntegrateKernel  gravity, integration, rotation, walls: one thread per body
//   2. GridKernel       the uniform grid by counting sort (one work-group); cell lists ordered by body index
//   3. NarrowKernel     contact detection, ALL bodies at the same time, one wave per body: the lanes scan the 27
//                       neighbour cells, then work out the contacts of the body's pairs (for cones / cylinders the
//                       lanes share the up to 64 x 64 part pairs). Read only: it sees the bodies as they are after
//                       integration and writes a contact list per body.
//   4. SolveKernel x27  the sequential-impulse solver over those contacts, which is cheap. It has to respect an
//                       order, so parallelism comes from grid colouring like in the OpenMP version: the cells of one
//                       colour are >= 3 cells apart and never touch the same body. One thread per cell.
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
constexpr int kCand = 32;         // partners remembered per neighbour cell while scanning
constexpr int kMaxPairs = 64;     // pairs a body can own (partners beyond that are ignored: never seen in practice)

// Contact of a pair as found by the narrow phase. The solver moves the bodies while it works through the pairs, so
// the contact is stored relative to them: the point relative to body a, and the separation along the normal at
// detection time, from which the current depth follows.
struct PairContact {
    float nx, ny, nz, depth;   // depth <= 0: no contact
    float rx, ry, rz, sep;     // contact point - a.pos; Dot(n, b.pos - a.pos) at detection
};
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
    int* pairCount = nullptr;    // per body: number of pairs it owns (partners with a larger index)
    int* pairBody = nullptr;     // per body: kMaxPairs partner indices
    PairContact* contacts = nullptr;   // OBB: the contact of each of those pairs
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

// ---- substep part 2: contact detection, one wave (work-group) per body, all bodies in parallel ----
// Body i owns its pairs with the bodies j > i of the 27 cells around it. The lanes scan one neighbour cell each (a
// long chain of dependent loads when one lane does it all), the hits are packed into the body's pair list in lane
// order, so the list does not depend on timing. AABB stops there (its contact test is trivial and is done by the
// solver). OBB works out the contacts: cheap pairs (spheres, cubes) one per lane, pairs with many part pairs
// (cones, cylinders) one after another with all the lanes cooperating.
__global__ __launch_bounds__(obb::kWave) void NarrowKernel(const Body* bodies, int obbMode, Grid g,
                                                           const int* cellCount, const int* cellStart,
                                                           const int* sorted, const int* cellOfBody, int* pairCount,
                                                           int* pairBody, PairContact* contacts) {
    // shared storage of the OBB part pairs (raw bytes, Box has a constructor)
    __shared__ char partStore[obb::WaveScratch::kBytes] __attribute__((aligned(16)));
    __shared__ int candCount[obb::kWave];
    __shared__ int cand[27][kCand];
    __shared__ int list[kMaxPairs];
    obb::WaveScratch scratch = obb::WaveScratch::Carve(partStore);

    const int lane = threadIdx.x;
    const int i = sorted[blockIdx.x];   // in cell order: neighbouring waves read the same memory
    const int own = cellOfBody[i];
    const int cx = own % g.nx, cy = own / g.nx % g.ny, cz = own / (g.nx * g.ny);
    const Body mine = bodies[i];

    // 1. gather the partners, one neighbour cell per lane
    int found = 0;
    if (lane < 27) {
        const int x = cx + lane % 3 - 1, y = cy + lane / 3 % 3 - 1, z = cz + lane / 9 - 1;
        if (x >= 0 && y >= 0 && z >= 0 && x < g.nx && y < g.ny && z < g.nz) {
            const int c = (z * g.ny + y) * g.nx + x;
            const int* cell = sorted + cellStart[c];
            const int cnt = cellCount[c];
            for (int ja = 0; ja < cnt && found < kCand; ++ja) {
                const int j = cell[ja];
                if (j > i && CouldTouch(mine, bodies[j], obbMode)) cand[lane][found++] = j;
            }
        }
    }
    candCount[lane] = found;
    if (obbMode) obb::PrepareA(mine, scratch);   // the parts of this body, once for all its pairs
    __syncthreads();

    // 2. pack them in lane order
    int before = 0, total = 0;
    for (int l = 0; l < 27; ++l) {
        if (l < lane) before += candCount[l];
        total += candCount[l];
    }
    total = min(total, kMaxPairs);
    for (int m = 0; m < found && before + m < kMaxPairs; ++m) {
        list[before + m] = cand[lane][m];
        pairBody[i * kMaxPairs + before + m] = cand[lane][m];
    }
    if (lane == 0) pairCount[i] = total;
    if (!obbMode) return;
    __syncthreads();

    // 3. the contacts
    PairContact* out = contacts + i * kMaxPairs;
    auto pack = [&](const Body& b, const obb::Contact& c, bool hit) {
        PairContact pc;
        pc.nx = c.n.x; pc.ny = c.n.y; pc.nz = c.n.z;
        pc.depth = hit ? c.depth : 0.0f;
        const Vec3 r = c.p - mine.pos;
        pc.rx = r.x; pc.ry = r.y; pc.rz = r.z;
        pc.sep = Dot(c.n, b.pos - mine.pos);
        return pc;
    };
    const int na = obb::PartCount(mine);
    bool anyBig = false;
    for (int k = lane; k < total; k += obb::kWave) {   // cheap pairs: one per lane
        const Body b = bodies[list[k]];
        if (na * obb::PartCount(b) > obb::kSmallTotal) {
            anyBig = true;
            continue;
        }
        obb::Contact c;
        const bool hit = ObbContact(mine, b, c);
        out[k] = pack(b, c, hit);
    }
    if (__ballot(anyBig) == 0) return;
    for (int k = 0; k < total; ++k) {   // pairs with many part pairs: the whole wave on one pair
        const Body b = bodies[list[k]];
        if (na * obb::PartCount(b) <= obb::kSmallTotal) continue;
        obb::Contact c;
        const bool hit = obb::ObbContactWave(mine, b, scratch, c);
        if (lane == 0) out[k] = pack(b, c, hit);
    }
}

// ---- substep part 3: the solver for the cells of one colour, one thread per cell ----
// Walks the pairs owned by the bodies of the cell in a fixed order and applies positional correction and impulses.
// The partners are in the 27 cells around, which no other cell of this colour touches.
// `reverse` walks the bodies and their pairs backwards (see LaunchSolve).
__global__ void SolveKernel(Body* bodies, int obbMode, Grid g, int color, int reverse, const int* cellCount,
                            const int* cellStart,
                            const int* sorted, const int* pairCount, const int* pairBody, const PairContact* contacts) {
    const int ox = color % 3, oy = (color / 3) % 3, oz = color / 9;
    const int cxn = (g.nx - ox + 2) / 3, cyn = (g.ny - oy + 2) / 3, czn = (g.nz - oz + 2) / 3;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cxn * cyn * czn) return;
    const int cx = ox + (idx % cxn) * 3;
    const int cy = oy + ((idx / cxn) % cyn) * 3;
    const int cz = oz + (idx / (cxn * cyn)) * 3;
    const int own = (cz * g.ny + cy) * g.nx + cx;

    const int first = cellStart[own], cnt = cellCount[own];
    for (int ib = 0; ib < cnt; ++ib) {
        const int i = sorted[first + (reverse ? cnt - 1 - ib : ib)];
        const int pairs = pairCount[i];
        if (pairs == 0) continue;
        Body a = bodies[i];
        for (int kb = 0; kb < pairs; ++kb) {
            const int k = reverse ? pairs - 1 - kb : kb;
            const int j = pairBody[i * kMaxPairs + k];
            if (obbMode) {
                const PairContact c = contacts[i * kMaxPairs + k];
                if (c.depth <= 0) continue;
                Body b = bodies[j];
                // earlier pairs of this sweep may have moved the two bodies: the depth follows the separation
                const Vec3 n(c.nx, c.ny, c.nz);
                const float depth = c.depth + c.sep - Dot(n, b.pos - a.pos);
                if (depth <= 0) continue;
                const Vec3 p = a.pos + Vec3(c.rx, c.ry, c.rz);
                SeparateBodies(a, b, n, depth);
                obb::ContactImpulse(a, b, n, p);
                bodies[j] = b;
            } else {
                Body b = bodies[j];
                AabbResolve(a, b);
                bodies[j] = b;
            }
        }
        bodies[i] = a;
    }
}

// ---- queueing one substep (asynchronous) ----
void LaunchGrid(Sim* s, float h, int obbMode) {
    IntegrateKernel<<<(s->count + 63) / 64, 64>>>(s->bodies, s->count, h, obbMode);
    GridKernel<<<1, kGridBlock>>>(s->bodies, s->count, GridFor(obbMode != 0), s->cellCount, s->cellStart,
                                  s->cellCursor, s->cellOfBody, s->sorted);
}

void LaunchNarrow(Sim* s, int obbMode) {
    NarrowKernel<<<s->count, obb::kWave>>>(s->bodies, obbMode, GridFor(obbMode != 0), s->cellCount, s->cellStart,
                                           s->sorted, s->cellOfBody, s->pairCount, s->pairBody, s->contacts);
}

// Every other substep sweeps backwards (colours, bodies, pairs): always going the same way pushes the bodies a
// little in one direction on average.
void LaunchSolve(Sim* s, int obbMode, int reverse) {
    const Grid g = GridFor(obbMode != 0);
    for (int pass = 0; pass < kColors; ++pass) {
        const int color = reverse ? kColors - 1 - pass : pass;
        const int ox = color % 3, oy = color / 3 % 3, oz = color / 9;
        const int cells = ((g.nx - ox + 2) / 3) * ((g.ny - oy + 2) / 3) * ((g.nz - oz + 2) / 3);
        SolveKernel<<<(cells + 63) / 64, 64>>>(s->bodies, obbMode, g, color, reverse, s->cellCount, s->cellStart, s->sorted,
                                               s->pairCount, s->pairBody, s->contacts);
    }
}

void LaunchSubStep(Sim* s, float h, int obbMode, int reverse) {
    LaunchGrid(s, h, obbMode);
    LaunchNarrow(s, obbMode);
    LaunchSolve(s, obbMode, reverse);
}

void FreeAll(Sim* s) {
    Free(s->bodies); Free(s->cellCount); Free(s->cellStart); Free(s->cellCursor);
    Free(s->cellOfBody); Free(s->sorted); Free(s->pairCount); Free(s->pairBody); Free(s->contacts);
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
              Check(hipMalloc(&s->pairCount, sizeof(int) * capacity), "hipMalloc pairCount") &&
              Check(hipMalloc(&s->pairBody, sizeof(int) * capacity * kMaxPairs), "hipMalloc pairBody") &&
              Check(hipMalloc(&s->contacts, sizeof(PairContact) * capacity * kMaxPairs), "hipMalloc contacts") &&
              Check(hipEventCreate(&s->start), "hipEventCreate") && Check(hipEventCreate(&s->stop), "hipEventCreate") &&
              Check(hipDeviceGetName(s->name, sizeof s->name, 0), "hipDeviceGetName");
    if (!ok) {
        FreeAll(s);
        delete s;
        return nullptr;
    }
    if (nameOut && nameLen > 0) std::snprintf(nameOut, nameLen, "%s", s->name);
    return s;
}

PH_EXPORT void ph_destroy(void* handle) {
    Sim* s = static_cast<Sim*>(handle);
    if (!s) return;
    FreeAll(s);
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
        LaunchSubStep(s, h, obbMode, step & 1);
    }
    if (!Check(hipGetLastError(), "kernel launch") || !Check(hipEventRecord(s->stop), "hipEventRecord") ||
        !Check(hipEventSynchronize(s->stop), "kernel execution"))
        return 0;
    float ms = 0;
    if (!Check(hipEventElapsedTime(&ms, s->start, s->stop), "hipEventElapsedTime")) return 0;
    if (gpuMs) *gpuMs = ms;
    return 1;
}

// Profiling aid (not used by the app): like ph_step, but reports how long the three phases took in total.
PH_EXPORT int ph_profile(void* handle, float h, int substeps, int obbMode, float* gridMs, float* narrowMs,
                         float* solveMs) {
    Sim* s = static_cast<Sim*>(handle);
    *gridMs = *narrowMs = *solveMs = 0;
    if (s->count == 0 || substeps < 1 || substeps > 4) return 0;
    hipEvent_t ev[13];
    for (auto& e : ev)
        if (!Check(hipEventCreate(&e), "hipEventCreate")) return 0;
    (void)hipEventRecord(ev[0]);
    for (int step = 0; step < substeps; ++step) {
        LaunchGrid(s, h, obbMode);
        (void)hipEventRecord(ev[3 * step + 1]);
        LaunchNarrow(s, obbMode);
        (void)hipEventRecord(ev[3 * step + 2]);
        LaunchSolve(s, obbMode, step & 1);
        (void)hipEventRecord(ev[3 * step + 3]);
    }
    bool ok = Check(hipEventSynchronize(ev[3 * substeps]), "profile");
    for (int step = 0; ok && step < substeps; ++step) {
        float g = 0, n = 0, p = 0;
        (void)hipEventElapsedTime(&g, ev[3 * step], ev[3 * step + 1]);
        (void)hipEventElapsedTime(&n, ev[3 * step + 1], ev[3 * step + 2]);
        (void)hipEventElapsedTime(&p, ev[3 * step + 2], ev[3 * step + 3]);
        *gridMs += g;
        *narrowMs += n;
        *solveMs += p;
    }
    for (auto& e : ev) (void)hipEventDestroy(e);
    return ok;
}

PH_EXPORT int ph_download(void* handle, Body* bodies, int n) {
    Sim* s = static_cast<Sim*>(handle);
    return n == 0 || Check(hipMemcpy(bodies, s->bodies, sizeof(Body) * n, hipMemcpyDeviceToHost), "download");
}
}
