#include "importer.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

namespace
{
Material ConvertMaterial(const aiMaterial& source)
{
    Material material;

    aiString name;
    if (source.Get(AI_MATKEY_NAME, name) == aiReturn_SUCCESS)
        material.name = name.C_Str();

    aiColor3D diffuse;
    if (source.Get(AI_MATKEY_COLOR_DIFFUSE, diffuse) == aiReturn_SUCCESS)
        material.diffuse = { diffuse.r, diffuse.g, diffuse.b };

    aiString texture;
    if (source.GetTexture(aiTextureType_DIFFUSE, 0, &texture) == aiReturn_SUCCESS)
        material.diffuseTexture = texture.C_Str();

    return material;
}
}  // namespace

bool ImportMesh(const std::string& path, Mesh& mesh)
{
    // ConvertToLeftHanded also flips the winding order and the V coordinate to match Direct3D.
    const unsigned flags = aiProcess_Triangulate | aiProcess_JoinIdenticalVertices |
                           aiProcess_GenSmoothNormals | aiProcess_PreTransformVertices |
                           aiProcess_ConvertToLeftHanded;

    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, flags);
    if (!scene || !scene->HasMeshes())
        return false;

    mesh = {};

    for (unsigned i = 0; i < scene->mNumMaterials; i++)
        mesh.materials.push_back(ConvertMaterial(*scene->mMaterials[i]));

    for (unsigned i = 0; i < scene->mNumMeshes; i++)
    {
        const aiMesh& source = *scene->mMeshes[i];
        const uint32_t firstVertex = static_cast<uint32_t>(mesh.vertices.size());

        SubMesh subMesh;
        subMesh.firstIndex = static_cast<uint32_t>(mesh.indices.size());
        subMesh.materialIndex = source.mMaterialIndex;

        for (unsigned v = 0; v < source.mNumVertices; v++)
        {
            Vertex vertex;
            vertex.position = { source.mVertices[v].x, source.mVertices[v].y, source.mVertices[v].z };
            vertex.normal = { source.mNormals[v].x, source.mNormals[v].y, source.mNormals[v].z };
            if (source.HasTextureCoords(0))
                vertex.uv = { source.mTextureCoords[0][v].x, source.mTextureCoords[0][v].y };
            mesh.vertices.push_back(vertex);
        }

        for (unsigned f = 0; f < source.mNumFaces; f++)
            for (unsigned k = 0; k < 3; k++)
                mesh.indices.push_back(firstVertex + source.mFaces[f].mIndices[k]);

        subMesh.indexCount = static_cast<uint32_t>(mesh.indices.size()) - subMesh.firstIndex;
        mesh.subMeshes.push_back(subMesh);
    }
    return true;
}
