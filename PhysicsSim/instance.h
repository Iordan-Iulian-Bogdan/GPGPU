#pragma once

#include "math3d.h"

// Per-instance data for drawing: one entry per simulated object.
struct Instance
{
    Vec4 position;  // xyz = center
    Vec4 scale;     // xyz = half extents (sphere: radius in all three)
    Vec4 color;
};
