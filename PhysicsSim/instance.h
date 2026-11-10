#pragma once

#include "math3d.h"

// Per-instance data for drawing: one entry per simulated object.
struct Instance
{
    Vec4 position;  // xyz = center
    Vec4 scale;     // xyz = half extents (sphere: radius in all three)
    Vec4 color;
    Vec4 axisX = { 1, 0, 0, 0 };  // orientation: the body's local axes in world space (identity = no rotation)
    Vec4 axisY = { 0, 1, 0, 0 };
    Vec4 axisZ = { 0, 0, 1, 0 };
};
