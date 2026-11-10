#include "camera.h"

#include <algorithm>

Vec3 Camera::Forward() const
{
    return { std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw) };
}

Vec3 Camera::Right() const
{
    return { std::cos(yaw), 0, -std::sin(yaw) };
}

Mat4 Camera::ViewMatrix() const
{
    return LookAt(position, position + Forward(), { 0, 1, 0 });
}

void Camera::Update(const Input& input, float dt)
{
    if (input.IsDown(MouseButton::Right))
    {
        yaw += input.MouseDeltaX() * lookSensitivity;
        pitch -= input.MouseDeltaY() * lookSensitivity;
        const float limit = ToRadians(89.0f);
        pitch = std::clamp(pitch, -limit, limit);
    }

    moveSpeed = std::clamp(moveSpeed * std::pow(1.1f, static_cast<float>(input.WheelNotches())), 0.5f, 100.0f);

    Vec3 direction = {};
    if (input.IsDown('W')) direction += Forward();
    if (input.IsDown('S')) direction += Forward() * -1.0f;
    if (input.IsDown('D')) direction += Right();
    if (input.IsDown('A')) direction += Right() * -1.0f;
    if (input.IsDown(VK_SPACE)) direction += Vec3{ 0, 1, 0 };
    if (input.IsDown(VK_CONTROL)) direction += Vec3{ 0, -1, 0 };

    float speed = input.IsDown(VK_SHIFT) ? moveSpeed * 3.0f : moveSpeed;
    position += Normalize(direction) * (speed * dt);
}
