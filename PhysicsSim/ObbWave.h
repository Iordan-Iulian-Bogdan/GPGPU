#pragma once
// GPU only (HIP): ObbResolve for a whole wave. All lanes of the wave call it with the SAME pair of bodies. The part
// pairs of a cone / cylinder (up to 64 x 64) are tested by all lanes together and only the overlapping ones are
// kept (in a deterministic order), then the lanes work out those contacts at once and combine the results with
// shuffles. Same result as ObbResolve up to the order of the floating point sums.
//
// Everything is computed in a frame whose origin is the centre of body a, so the parts of a (which do not change
// while a is being swept against its neighbours) are generated only once per sweep: PrepareA.
#include <hip/hip_runtime.h>

#include "ObbCollision.h"

namespace phys {
namespace obb {
inline constexpr int kWave = 32;           // lanes that cooperate on one pair (the work-group size)
inline constexpr int kSmallTotal = 32;     // up to this many part pairs every lane just does the scalar version
inline constexpr int kMaxHits = 512;       // overlapping part pairs kept per body pair (the rest is ignored)

struct HitData {
    float nx, ny, nz, depth;               // contact normal and depth of one overlapping part pair
};

// Shared memory of one wave, carved out of raw bytes (Box has a constructor, which __shared__ does not allow).
struct WaveScratch {
    Box* a;        // parts of body a, relative to a's centre
    Box* b;        // parts of body b, relative to a's centre
    float* ra;     // bounding sphere radius of every part
    float* rb;
    int* hitK;     // overlapping part pairs: (part of a << 8) | part of b
    HitData* hit;  // their contacts

    static constexpr size_t kBytes = 2 * kMaxParts * sizeof(Box) + 2 * kMaxParts * sizeof(float) +
                                     kMaxHits * sizeof(int) + kMaxHits * sizeof(HitData);

    __device__ static WaveScratch Carve(char* mem) {
        WaveScratch s;
        s.a = reinterpret_cast<Box*>(mem);
        s.b = s.a + kMaxParts;
        s.ra = reinterpret_cast<float*>(s.b + kMaxParts);
        s.rb = s.ra + kMaxParts;
        s.hitK = reinterpret_cast<int*>(s.rb + kMaxParts);
        s.hit = reinterpret_cast<HitData*>(s.hitK + kMaxHits);
        return s;
    }
};

__device__ inline float WaveSum(float v) {
    for (int o = kWave / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, kWave);
    return v;   // identical in every lane (a + b == b + a at every butterfly stage)
}
__device__ inline float WaveMax(float v) {
    for (int o = kWave / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, kWave));
    return v;
}
__device__ inline int WaveMin(int v) {
    for (int o = kWave / 2; o > 0; o >>= 1) v = min(v, __shfl_xor(v, o, kWave));
    return v;
}

// cos(pi * m / 8) for m in [-8, 8]
__device__ inline float Cos8(int m) {
    m = m < 0 ? -m : m;
    float sign = 1.0f;
    if (m > 4) { m = 8 - m; sign = -1.0f; }
    const float v = m == 0 ? 1.0f : (m == 1 ? 0.92387953f : (m == 2 ? 0.70710678f : (m == 3 ? 0.38268343f : 0.0f)));
    return sign * v;
}

// Part k of a body whose centre is at `centre` (in whatever frame the caller works in) and the radius of the part's
// bounding sphere. Same boxes as PartAt / MakeParts, but without trigonometry: the bars of a cylinder fan out in
// steps of 22.5 degrees, and a bar's half width is rl sin(pi/16), its half length rl cos(pi/16).
__device__ inline void LocalPart(const Body& body, int k, Vec3 centre, Box& box, float& radius) {
    if (body.shape != Shape::Cone && body.shape != Shape::Cylinder) {
        box = ToBox(body);
        box.pos = centre;
        radius = Length(body.half);
        return;
    }
    const float r = body.half.x, h = body.half.y;
    float yc = 0, rl = r, hh = h;
    int bar = k;
    if (body.shape == Shape::Cone) {
        const float layerH = h / kConeLayers;
        const int ly = k / kCylinderBars;
        bar = k % kCylinderBars;
        yc = -h + layerH * (2 * ly + 1);
        rl = r * (h - yc) / (2 * h);
        hh = layerH;
    }
    const float cs = Cos8(bar), sn = Cos8(4 - bar);   // cos / sin of pi * bar / 8
    box.pos = centre + body.ax[1] * yc;
    box.half = {rl * 0.98078528f, hh, rl * 0.19509032f};
    box.ax[0] = body.ax[0] * cs + body.ax[2] * sn;
    box.ax[1] = body.ax[1];
    box.ax[2] = body.ax[2] * cs - body.ax[0] * sn;
    box.sphere = false;
    radius = sqrtf(rl * rl + hh * hh);
}

// The parts of body a, once per sweep (a's orientation does not change while its pairs are resolved). All lanes
// call it; a barrier is needed before ObbResolveWave uses them.
__device__ inline void PrepareA(const Body& a, WaveScratch& s) {
    const int lane = threadIdx.x % kWave, na = PartCount(a);
    for (int i = lane; i < na; i += kWave) LocalPart(a, i, Vec3(0, 0, 0), s.a[i], s.ra[i]);
}

__device__ inline void ObbResolveWave(Body& a, Body& b, WaveScratch& s) {
    // cheap reject: bounding spheres (radius = half diagonal) must overlap first
    const Vec3 d = b.pos - a.pos;
    const float rr = Length(a.half) + Length(b.half);
    if (Dot(d, d) >= rr * rr) return;

    const int na = PartCount(a), nb = PartCount(b);
    if (na * nb <= kSmallTotal) {   // sphere / cube / cylinder-cube...: not worth sharing, all lanes do the same
        ObbResolve(a, b);
        return;
    }

    const int lane = threadIdx.x % kWave;
    __syncthreads();   // the previous pair may still be reading s.b / s.hit
    for (int j = lane; j < nb; j += kWave) LocalPart(b, j, d, s.b[j], s.rb[j]);
    __syncthreads();

    // 1. find the part pairs whose bounding spheres overlap; the lanes test one pair each and the hits are packed in
    //    pair order with a ballot, so the result does not depend on timing
    int nh = 0;
    for (int i = 0; i < na; ++i)
        for (int jb = 0; jb < nb; jb += kWave) {
            const int j = jb + lane;
            bool pass = false;
            if (j < nb) {
                const Vec3 pd = s.b[j].pos - s.a[i].pos;
                const float pr = s.ra[i] + s.rb[j];
                pass = Dot(pd, pd) < pr * pr;
            }
            const unsigned long long mask = __ballot(pass);
            if (pass) {
                const int slot = nh + __builtin_popcountll(mask & ((1ull << lane) - 1));
                if (slot < kMaxHits) s.hitK[slot] = (i << 8) | j;
            }
            nh += __builtin_popcountll(mask);
        }
    nh = min(nh, kMaxHits);
    __syncthreads();

    // 2. the contacts of those pairs, spread over the lanes: depth-weighted sums of normal and point
    Vec3 nSum(0, 0, 0), pSum(0, 0, 0);
    float wSum = 0;
    int firstHit = 1 << 30;
    for (int h = lane; h < nh; h += kWave) {
        const int i = s.hitK[h] >> 8, j = s.hitK[h] & 255;
        Contact t;
        if (Narrow(s.a[i], s.b[j], t)) {
            nSum += t.n * t.depth;
            pSum += t.p * t.depth;
            wSum += t.depth;
            s.hit[h] = {t.n.x, t.n.y, t.n.z, t.depth};
            firstHit = min(firstHit, h);
        } else {
            s.hit[h].depth = 0;
        }
    }
    nSum = {WaveSum(nSum.x), WaveSum(nSum.y), WaveSum(nSum.z)};
    pSum = {WaveSum(pSum.x), WaveSum(pSum.y), WaveSum(pSum.z)};
    wSum = WaveSum(wSum);
    if (wSum < 1e-9f) return;   // no real overlap (same guard as ObbResolve)
    __syncthreads();            // the hit data written by the other lanes is read below

    Contact c;
    const float nl = Length(nSum);
    if (nl > 1e-6f) {
        c.n = nSum * (1.0f / nl);
    } else {   // opposing normals cancelled: use the first overlapping pair's normal
        const HitData& t = s.hit[WaveMin(firstHit)];
        c.n = Vec3(t.nx, t.ny, t.nz);
    }
    c.p = pSum * (1.0f / wSum) + a.pos;   // back from a's frame to the world

    // 3. the depth is the largest overlap of any pair measured along the final normal
    float depth = 0;
    for (int h = lane; h < nh; h += kWave) {
        const HitData& t = s.hit[h];
        if (t.depth > 0) depth = fmaxf(depth, t.depth * fmaxf(0.0f, t.nx * c.n.x + t.ny * c.n.y + t.nz * c.n.z));
    }
    c.depth = WaveMax(depth);

    // every lane applies the same response to the same bodies (identical values, so the writes agree)
    SeparateBodies(a, b, c.n, c.depth);
    ContactImpulse(a, b, c.n, c.p);
}
}  // namespace obb
}  // namespace phys
