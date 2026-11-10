#pragma once

#include "mesh.h"

#include <string>

// Imports any format Assimp knows (.obj, .fbx, .3ds, .x, ...) into the engine mesh format.
bool ImportMesh(const std::string& path, Mesh& mesh);
