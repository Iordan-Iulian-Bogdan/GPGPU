#pragma once
#include "Physics.h"

// Axis-aligned bounding boxes: every body collides as an upright box, bodies never rotate.
class AabbWorld : public World {
public:
    bool Obb() const override { return false; }

protected:
    void UpdateBody(Body& b, float h) override;   // wall collisions
    void Resolve(Body& a, Body& b) override;      // box-box test + linear impulse
    void OnAdopt(Body& b) override;               // AABBs cannot rotate: reset the orientation
};
