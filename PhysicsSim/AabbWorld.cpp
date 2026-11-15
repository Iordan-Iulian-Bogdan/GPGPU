// Axis-aligned-box collision model of the CPU world. The tests and the response are in AabbCollision.h,
// shared with the GPU (HIP) backend.
#include "AabbWorld.h"
#include "AabbCollision.h"

using namespace phys;

void AabbWorld::OnAdopt(Body& b) {
    b.ax[0] = {1, 0, 0};
    b.ax[1] = {0, 1, 0};
    b.ax[2] = {0, 0, 1};
    b.w = {};
}

void AabbWorld::UpdateBody(Body& b, float) { AabbUpdateBody(b); }

void AabbWorld::Resolve(Body& a, Body& b) { AabbResolve(a, b); }
