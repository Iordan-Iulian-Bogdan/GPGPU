// Common part of the simulation (see Physics.h). Collision models live in AabbWorld.cpp / ObbWorld.cpp.
#include "Physics.h"
#include <algorithm>
#include "PhysicsCommon.h"

using namespace phys;

namespace {
// Cell size must be >= the largest centre distance at which two bodies can touch: two bounding spheres of
// radius 0.61 (a rotated cube's half-diagonal) = 1.22.
constexpr float kCell = 1.25f;
constexpr int kSubSteps = 2;
constexpr int NX = (int)(2 * World::W / kCell) + 1;
constexpr int NY = (int)(World::H / kCell) + 1;
constexpr int NZ = (int)(2 * World::D / kCell) + 1;
}  // namespace

World::World() : rng_(12345) {
    bodies_.reserve(MaxBodies);
    head_.resize((size_t)NX * NY * NZ);
}

void World::Clear() { bodies_.clear(); }

void World::AdoptState(const World& other) {
    bodies_ = other.bodies_;
    rng_ = other.rng_;
    parallel_ = other.parallel_;
    for (Body& b : bodies_) OnAdopt(b);
}

void World::AddBody(Shape s, Vec3 pos, Vec3 vel) {
    if (bodies_.size() >= MaxBodies) return;
    Body b;
    b.shape = s;
    b.pos = pos;
    b.vel = vel;
    float mass = 1;
    Vec3 inertia;   // principal moments of inertia (unit density scale, like mass)
    switch (s) {
    case Shape::Sphere: {
        float r = Rand(0.2f, 0.4f);
        b.half = {r, r, r};
        mass = 4.0f / 3.0f * kPi * r * r * r;
        float i = 0.4f * mass * r * r;
        inertia = {i, i, i};
        break;
    }
    case Shape::Cube: {
        b.half = {Rand(0.2f, 0.35f), Rand(0.2f, 0.35f), Rand(0.2f, 0.35f)};
        mass = 8 * b.half.x * b.half.y * b.half.z;
        Vec3 e = b.half * 2.0f;   // full edge lengths
        inertia = {mass * (e.y * e.y + e.z * e.z) / 12, mass * (e.x * e.x + e.z * e.z) / 12,
                   mass * (e.x * e.x + e.y * e.y) / 12};
        break;
    }
    case Shape::Cylinder:
    case Shape::Cone: {
        float r = Rand(0.15f, 0.3f), h = Rand(0.2f, 0.4f);
        b.half = {r, h, r};     // bounding volume (AABB / OBB) used for collision
        float len = 2 * h;
        if (s == Shape::Cylinder) {
            mass = kPi * r * r * len;
            inertia = {mass * (3 * r * r + len * len) / 12, 0.5f * mass * r * r, mass * (3 * r * r + len * len) / 12};
        } else {
            mass = kPi * r * r * len / 3.0f;
            float ixz = 3.0f / 20.0f * mass * r * r + 3.0f / 80.0f * mass * len * len;   // about the centroid
            inertia = {ixz, 0.3f * mass * r * r, ixz};
        }
        break;
    }
    }
    b.invMass = 1.0f / (mass * 3.0f);   // density 3 -> keeps masses in a comfortable range
    b.invI = {1.0f / (inertia.x * 3.0f), 1.0f / (inertia.y * 3.0f), 1.0f / (inertia.z * 3.0f)};
    b.color = {Rand(0.25f, 1.0f), Rand(0.25f, 1.0f), Rand(0.25f, 1.0f)};
    InitBody(b);
    bodies_.push_back(b);
}

void World::Spawn(Shape s, bool atTop) {
    (void)atTop;
    Vec3 p(Rand(-W + 0.6f, W - 0.6f), Rand(H - 6.0f, H - 1.0f), Rand(-D + 0.6f, D - 0.6f));
    Vec3 v(Rand(-4, 4), Rand(-4, 2), Rand(-4, 4));
    AddBody(s, p, v);
}

void World::LoadPreset(int index) {
    static const int counts[3][3] = {{100, 250, 500}, {250, 500, 1000}, {500, 1000, 2500}};
    index = Clampi(index, 0, 2);
    Clear();

    // jittered lattice so the random initial positions never overlap
    const float s = 0.9f;
    std::vector<Vec3> pts;
    for (float y = 0.5f; y < H - 0.5f; y += s)
        for (float x = -W + 0.45f; x < W - 0.45f; x += s)
            for (float z = -D + 0.45f; z < D - 0.45f; z += s) pts.push_back({x, y, z});
    std::shuffle(pts.begin(), pts.end(), rng_);

    std::vector<Shape> kinds;
    for (int i = 0; i < counts[index][0]; ++i) kinds.push_back(Shape::Sphere);
    for (int i = 0; i < counts[index][1]; ++i) kinds.push_back(Shape::Cube);
    for (int i = 0; i < counts[index][2]; ++i) kinds.push_back(i % 2 ? Shape::Cone : Shape::Cylinder);

    for (size_t i = 0; i < kinds.size() && i < pts.size(); ++i) {
        Vec3 p = pts[i] + Vec3(Rand(-0.05f, 0.05f), Rand(-0.05f, 0.05f), Rand(-0.05f, 0.05f));
        AddBody(kinds[i], p, Vec3(Rand(-3, 3), Rand(-3, 3), Rand(-3, 3)));
    }
}

void World::Kick() {
    for (Body& b : bodies_) {
        b.vel += Vec3(Rand(-6, 6), Rand(3, 10), Rand(-6, 6));
        OnKick(b);
    }
}

void World::SeparateBodies(Body& a, Body& b, Vec3 n, float depth) {
    float sum = a.invMass + b.invMass;
    float corr = std::max(depth - kSlop, 0.0f) * kCorrection / sum;
    corr = std::min(corr, 0.5f);
    a.pos -= n * (corr * a.invMass);
    b.pos += n * (corr * b.invMass);
}

void World::SubStep(float h) {
    // gravity + integration + walls (cheap, serial)
    for (Body& b : bodies_) {
        b.vel.y -= kGravity * h;
        b.pos += b.vel * h;
        UpdateBody(b, h);
    }

    // broad phase: uniform grid (linked lists)
    std::fill(head_.begin(), head_.end(), -1);
    next_.assign(bodies_.size(), -1);
    auto cellOf = [](const Vec3& p, int& cx, int& cy, int& cz) {
        cx = Clampi((int)((p.x + World::W) / kCell), 0, NX - 1);
        cy = Clampi((int)(p.y / kCell), 0, NY - 1);
        cz = Clampi((int)((p.z + World::D) / kCell), 0, NZ - 1);
    };
    for (int i = 0; i < (int)bodies_.size(); ++i) {
        int cx, cy, cz;
        cellOf(bodies_[i].pos, cx, cy, cz);
        int c = (cz * NY + cy) * NX + cx;
        next_[i] = head_[c];
        head_[c] = i;
    }

    // narrow phase + response (sequential-impulse solver, each pair once: j > i).
    // Parallelised by grid colouring: a body only touches bodies in the 3x3x3 cells around its own
    // cell, so two cells that are >= 3 apart on some axis never share a body. Cells are grouped by
    // (x%3, y%3, z%3) into 27 colours; the cells of one colour run in parallel, colours run one
    // after another. No locks or atomics needed, and the result does not depend on thread timing.
    for (int color = 0; color < 27; ++color) {
        const int ox = color % 3, oy = (color / 3) % 3, oz = color / 9;
        const int cxn = (NX - ox + 2) / 3, cyn = (NY - oy + 2) / 3, czn = (NZ - oz + 2) / 3;
        const int cells = cxn * cyn * czn;

        #pragma omp parallel for schedule(dynamic, 4) if (parallel_)
        for (int idx = 0; idx < cells; ++idx) {
            const int cx = ox + (idx % cxn) * 3;
            const int cy = oy + ((idx / cxn) % cyn) * 3;
            const int cz = oz + (idx / (cxn * cyn)) * 3;
            for (int i = head_[(cz * NY + cy) * NX + cx]; i >= 0; i = next_[i])
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            int x = cx + dx, y = cy + dy, z = cz + dz;
                            if (x < 0 || y < 0 || z < 0 || x >= NX || y >= NY || z >= NZ) continue;
                            for (int j = head_[(z * NY + y) * NX + x]; j >= 0; j = next_[j])
                                if (j > i) Resolve(bodies_[i], bodies_[j]);
                        }
        }
    }
}

void World::Step(float dt) {
    float h = dt / kSubSteps;
    for (int s = 0; s < kSubSteps; ++s) SubStep(h);
}

void World::BuildInstances(std::vector<Instance> out[4]) const {
    for (int i = 0; i < 4; ++i) out[i].clear();

    // the five visible walls (front face z=-D is omitted on purpose)
    const float t = 0.25f, gray[4] = {0.45f, 0.47f, 0.52f, 1};
    auto wall = [&](Vec3 c, Vec3 hs) {
        Instance in{{c.x, c.y, c.z, 0}, {hs.x, hs.y, hs.z, 0}, {gray[0], gray[1], gray[2], 1}};
        out[(int)Shape::Cube].push_back(in);
    };
    wall({0, -t, 0}, {W + 2 * t, t, D + 2 * t});               // floor
    wall({0, H / 2, D + t}, {W + 2 * t, H / 2, t});            // back
    wall({-W - t, H / 2, 0}, {t, H / 2, D});                   // left
    wall({W + t, H / 2, 0}, {t, H / 2, D});                    // right

    for (const Body& b : bodies_) {
        Instance in{{b.pos.x, b.pos.y, b.pos.z, 0}, {b.half.x, b.half.y, b.half.z, 0},
                    {b.color.x, b.color.y, b.color.z, 1}};
        in.axisX = {b.ax[0].x, b.ax[0].y, b.ax[0].z, 0};
        in.axisY = {b.ax[1].x, b.ax[1].y, b.ax[1].z, 0};
        in.axisZ = {b.ax[2].x, b.ax[2].y, b.ax[2].z, 0};
        out[(int)b.shape].push_back(in);
    }
}
