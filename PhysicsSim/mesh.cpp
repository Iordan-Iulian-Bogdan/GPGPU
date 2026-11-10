#include "mesh.h"

#include <fstream>

namespace
{
constexpr uint32_t Magic = 0x4433474D;  // "MG3D"
constexpr uint32_t Version = 1;

template <typename T>
void Write(std::ofstream& out, const T& value)
{
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
bool Read(std::ifstream& in, T& value)
{
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

template <typename T>
void WriteVector(std::ofstream& out, const std::vector<T>& v)
{
    Write(out, static_cast<uint32_t>(v.size()));
    out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}

template <typename T>
bool ReadVector(std::ifstream& in, std::vector<T>& v)
{
    uint32_t count = 0;
    if (!Read(in, count))
        return false;
    v.resize(count);
    return static_cast<bool>(in.read(reinterpret_cast<char*>(v.data()), count * sizeof(T)));
}

void WriteString(std::ofstream& out, const std::string& s)
{
    Write(out, static_cast<uint32_t>(s.size()));
    out.write(s.data(), s.size());
}

bool ReadString(std::ifstream& in, std::string& s)
{
    uint32_t size = 0;
    if (!Read(in, size))
        return false;
    s.resize(size);
    return static_cast<bool>(in.read(s.data(), size));
}
}  // namespace

bool SaveMesh(const Mesh& mesh, const std::string& path)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
        return false;

    Write(out, Magic);
    Write(out, Version);
    WriteVector(out, mesh.vertices);
    WriteVector(out, mesh.indices);
    WriteVector(out, mesh.subMeshes);

    Write(out, static_cast<uint32_t>(mesh.materials.size()));
    for (const Material& material : mesh.materials)
    {
        WriteString(out, material.name);
        Write(out, material.diffuse);
        WriteString(out, material.diffuseTexture);
    }
    return static_cast<bool>(out);
}

bool LoadMesh(const std::string& path, Mesh& mesh)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;

    uint32_t magic = 0, version = 0;
    if (!Read(in, magic) || !Read(in, version) || magic != Magic || version != Version)
        return false;

    mesh = {};
    if (!ReadVector(in, mesh.vertices) || !ReadVector(in, mesh.indices) || !ReadVector(in, mesh.subMeshes))
        return false;

    uint32_t materialCount = 0;
    if (!Read(in, materialCount))
        return false;
    mesh.materials.resize(materialCount);
    for (Material& material : mesh.materials)
    {
        if (!ReadString(in, material.name) || !Read(in, material.diffuse) ||
            !ReadString(in, material.diffuseTexture))
            return false;
    }
    return true;
}
