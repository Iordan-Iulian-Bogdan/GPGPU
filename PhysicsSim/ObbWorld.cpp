// Oriented-box collision model of the CPU world. The tests and the response are in ObbCollision.h, shared
// with the GPU (HIP) backend.
#include "ObbWorld.h"
#include "ObbCollision.h"

using namespace phys;

void ObbWorld::InitBody(Body& b) { b.w = {Rand(-3, 3), Rand(-3, 3), Rand(-3, 3)}; }

void ObbWorld::OnKick(Body& b) { b.w += Vec3(Rand(-8, 8), Rand(-8, 8), Rand(-8, 8)); }

void ObbWorld::OnAdopt(Body& b) { InitBody(b); }   // visible spin right away

void ObbWorld::UpdateBody(Body& b, float h) { ObbUpdateBody(b, h); }

void ObbWorld::Resolve(Body& a, Body& b) { ObbResolve(a, b); }
