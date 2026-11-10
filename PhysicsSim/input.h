#pragma once

#include <Windows.h>

#include <bitset>

enum class MouseButton { Left, Right, Middle };

// Polled input state. The window procedure feeds events in, the main loop calls NewFrame()
// at the end of each frame to reset the per-frame values.
class Input
{
public:
    void OnKey(UINT virtualKey, bool down);
    void OnMouseButton(MouseButton button, bool down);
    void OnMouseMove(int dx, int dy);
    void OnWheel(int notches);

    bool IsDown(UINT virtualKey) const { return keyDown[virtualKey]; }
    bool WasPressed(UINT virtualKey) const { return keyPressed[virtualKey]; }
    bool WasReleased(UINT virtualKey) const { return keyReleased[virtualKey]; }

    bool IsDown(MouseButton button) const { return buttonDown[static_cast<int>(button)]; }
    bool WasPressed(MouseButton button) const { return buttonPressed[static_cast<int>(button)]; }
    bool WasReleased(MouseButton button) const { return buttonReleased[static_cast<int>(button)]; }

    int MouseDeltaX() const { return mouseDx; }
    int MouseDeltaY() const { return mouseDy; }
    int WheelNotches() const { return wheel; }

    void NewFrame();

private:
    std::bitset<256> keyDown, keyPressed, keyReleased;
    std::bitset<3> buttonDown, buttonPressed, buttonReleased;
    int mouseDx = 0, mouseDy = 0, wheel = 0;
};
