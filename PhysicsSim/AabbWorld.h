#pragma once
#include "Physics.h"

// Axis-aligned bounding boxes: cheap overlap tests, bodies never rotate.
class AabbWorld : public World {
public:
    bool Obb() const override { return false; }

protected:
    void UpdateBody(Body& b, float h) override;   // wall collisions
    void Resolve(Body& a, Body& b) override;      // AABB / sphere tests + linear impulse
    void OnAdopt(Body& b) override;               // AABBs cannot rotate: reset the orientation
};
