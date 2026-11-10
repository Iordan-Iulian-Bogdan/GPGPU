#pragma once

#include "input.h"
#include "math3d.h"

// First-person camera: WASD to move, Space/Ctrl up/down, Shift to run,
// mouse look while the right mouse button is held, wheel changes speed.
class Camera
{
public:
    Vec3 position = { 0, 1.5f, -4 };
    float yaw = 0;    // radians, 0 looks along +Z
    float pitch = 0;  // radians, positive looks up
    float moveSpeed = 4.0f;
    float lookSensitivity = 0.0025f;

    void Update(const Input& input, float dt);

    Vec3 Forward() const;
    Vec3 Right() const;
    Mat4 ViewMatrix() const;
};
