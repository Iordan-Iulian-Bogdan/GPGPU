#pragma once

#include <cmath>

// Conventions: left-handed, Y up, row vectors (v * M), matrices stored row-major.
// This matches Direct3D, so matrices can be sent to HLSL as row_major float4x4.

constexpr float Pi = 3.14159265358979f;

inline float ToRadians(float degrees) { return degrees * Pi / 180.0f; }

struct Vec2
{
    float x = 0, y = 0;
};

struct Vec3
{
    float x = 0, y = 0, z = 0;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline Vec3 operator*(Vec3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
inline Vec3 operator-(Vec3 a) { return { -a.x, -a.y, -a.z }; }
inline Vec3& operator+=(Vec3& a, Vec3 b) { a = a + b; return a; }
inline Vec3& operator-=(Vec3& a, Vec3 b) { a = a - b; return a; }

inline float Dot(Vec3 a, Vec3 b) 
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z; 
}

inline Vec3 Cross(Vec3 a, Vec3 b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

inline float Length(Vec3 v) { return std::sqrt(Dot(v, v)); }

inline Vec3 Normalize(Vec3 v)
{
    float len = Length(v);
    return len > 0 ? v * (1.0f / len) : Vec3{};
}

struct Vec4
{
    float x = 0, y = 0, z = 0, w = 0;
};

struct Mat4
{
    float m[4][4] = {};
};

inline Mat4 Identity()
{
    Mat4 r;
    for (int i = 0; i < 4; i++)
        r.m[i][i] = 1;
    return r;
}

// With row vectors, (a * b) applies a first, then b.
inline Mat4 operator*(const Mat4& a, const Mat4& b)
{
    Mat4 r;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                r.m[i][j] += a.m[i][k] * b.m[k][j];
    return r;
}

inline Mat4 Translation(Vec3 t)
{
    Mat4 r = Identity();
    r.m[3][0] = t.x;
    r.m[3][1] = t.y;
    r.m[3][2] = t.z;
    return r;
}

inline Mat4 Scaling(Vec3 s)
{
    Mat4 r = Identity();
    r.m[0][0] = s.x;
    r.m[1][1] = s.y;
    r.m[2][2] = s.z;
    return r;
}

inline Mat4 RotationY(float radians)
{
    float c = std::cos(radians), s = std::sin(radians);
    Mat4 r = Identity();
    r.m[0][0] = c;  r.m[0][2] = -s;
    r.m[2][0] = s;  r.m[2][2] = c;
    return r;
}

inline Mat4 LookAt(Vec3 eye, Vec3 target, Vec3 up)
{
    Vec3 z = Normalize(target - eye);
    Vec3 x = Normalize(Cross(up, z));
    Vec3 y = Cross(z, x);

    Mat4 r = Identity();
    r.m[0][0] = x.x;  r.m[0][1] = y.x;  r.m[0][2] = z.x;
    r.m[1][0] = x.y;  r.m[1][1] = y.y;  r.m[1][2] = z.y;
    r.m[2][0] = x.z;  r.m[2][1] = y.z;  r.m[2][2] = z.z;
    r.m[3][0] = -Dot(x, eye);
    r.m[3][1] = -Dot(y, eye);
    r.m[3][2] = -Dot(z, eye);
    return r;
}

// Depth range 0..1 as in Direct3D.
inline Mat4 Perspective(float fovY, float aspect, float nearZ, float farZ)
{
    float yScale = 1.0f / std::tan(fovY * 0.5f);
    Mat4 r;
    r.m[0][0] = yScale / aspect;
    r.m[1][1] = yScale;
    r.m[2][2] = farZ / (farZ - nearZ);
    r.m[2][3] = 1;
    r.m[3][2] = -nearZ * farZ / (farZ - nearZ);
    return r;
}
