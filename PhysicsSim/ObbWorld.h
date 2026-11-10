#pragma once
#include "Physics.h"

// Oriented bounding boxes: SAT collision tests, rigid-body rotation (angular velocity, inertia, torque).
class ObbWorld : public World {
public:
    bool Obb() const override { return true; }

protected:
    void UpdateBody(Body& b, float h) override;   // spin damping + rotation + wall contacts
    void Resolve(Body& a, Body& b) override;      // OBB / sphere tests + impulses with torque
    void InitBody(Body& b) override;              // new bodies start spinning
    void OnKick(Body& b) override;                // kicks add spin
    void OnAdopt(Body& b) override;               // bodies taken over from AABB start spinning
};
