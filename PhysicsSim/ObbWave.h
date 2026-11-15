#pragma once
// GPU only (HIP): ObbResolve for a whole wave. All lanes of the wave call it with the SAME pair of bodies; the
// part pairs of a cone / cylinder (up to 64 x 64) are split over the lanes and the partial sums are combined
// with shuffles. Same result as ObbResolve up to the order of the floating point sums.
#include <hip/hip_runtime.h>

#include "ObbCollision.h"

namespace phys {
namespace obb {
inline constexpr int kWave = 32;           // lanes that cooperate on one pair (the work-group size)
inline constexpr int kSmallTotal = 32;     // up to this many part pairs every lane just does the scalar version

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

// sa / sb: shared memory for the parts of a and b (kMaxParts boxes each).
__device__ inline void ObbResolveWave(Body& a, Body& b, Box* sa, Box* sb) {
    // cheap reject: bounding spheres (radius = half diagonal) must overlap first
    const Vec3 d = b.pos - a.pos;
    const float rr = Length(a.half) + Length(b.half);
    if (Dot(d, d) >= rr * rr) return;

    const int na = PartCount(a), nb = PartCount(b), total = na * nb;
    if (total <= kSmallTotal) {   // sphere / cube / cylinder-cube...: not worth sharing, all lanes do the same
        ObbResolve(a, b);
        return;
    }

    // parts of both bodies into shared memory, a few per lane
    const int lane = threadIdx.x % kWave;
    __syncthreads();   // the previous pair may still be reading sa / sb
    for (int i = lane; i < na; i += kWave) sa[i] = PartAt(a, i);
    for (int j = lane; j < nb; j += kWave) sb[j] = PartAt(b, j);
    __syncthreads();

    // contact of part pair k, false if the parts do not overlap
    auto contact = [&](int k, Contact& t) {
        const int i = k / nb, j = k - i * nb;
        const Vec3 pd = sb[j].pos - sa[i].pos;
        const float pr = Length(sa[i].half) + Length(sb[j].half);
        if (Dot(pd, pd) >= pr * pr) return false;   // cheap per-part reject
        return Narrow(sa[i], sb[j], t);
    };

    // pass 1: depth-weighted sums of normal and point over all overlapping part pairs
    Vec3 nSum(0, 0, 0), pSum(0, 0, 0);
    float wSum = 0;
    int firstHit = 1 << 30;
    for (int k = lane; k < total; k += kWave) {
        Contact t;
        if (!contact(k, t)) continue;
        nSum += t.n * t.depth;
        pSum += t.p * t.depth;
        wSum += t.depth;
        firstHit = min(firstHit, k);
    }
    nSum = {WaveSum(nSum.x), WaveSum(nSum.y), WaveSum(nSum.z)};
    pSum = {WaveSum(pSum.x), WaveSum(pSum.y), WaveSum(pSum.z)};
    wSum = WaveSum(wSum);
    if (wSum < 1e-9f) return;   // no real overlap (same guard as ObbResolve)

    Contact c;
    const float nl = Length(nSum);
    if (nl > 1e-6f) {
        c.n = nSum * (1.0f / nl);
    } else {   // opposing normals cancelled: use the first overlapping pair's normal
        firstHit = WaveMin(firstHit);
        Contact t;
        contact(firstHit, t);
        c.n = t.n;
    }
    c.p = pSum * (1.0f / wSum);

    // pass 2: the depth is the largest overlap of any pair measured along the final normal
    float depth = 0;
    for (int k = lane; k < total; k += kWave) {
        Contact t;
        if (contact(k, t)) depth = fmaxf(depth, t.depth * fmaxf(0.0f, Dot(t.n, c.n)));
    }
    c.depth = WaveMax(depth);

    // every lane applies the same response to the same bodies (identical values, so the writes agree)
    SeparateBodies(a, b, c.n, c.depth);
    ContactImpulse(a, b, c.n, c.p);
}
}  // namespace obb
}  // namespace phys
