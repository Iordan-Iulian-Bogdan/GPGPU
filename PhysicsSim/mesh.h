#pragma once

#include "math3d.h"

#include <cstdint>
#include <string>
#include <vector>

// Engine-internal mesh format (.mg3dmesh): triangles, materials, texture references,
// normals and texture coordinates.

struct Vertex
{
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
};

struct Material
{
    std::string name;
    Vec3 diffuse = { 0.8f, 0.8f, 0.8f };
    std::string diffuseTexture;  // path, empty if none
};

// A range of triangles that share one material.
struct SubMesh
{
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    uint32_t materialIndex = 0;
};

struct Mesh
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Material> materials;
    std::vector<SubMesh> subMeshes;
};

bool SaveMesh(const Mesh& mesh, const std::string& path);
bool LoadMesh(const std::string& path, Mesh& mesh);
