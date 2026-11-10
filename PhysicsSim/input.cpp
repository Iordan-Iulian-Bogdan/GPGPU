#include "input.h"

void Input::OnKey(UINT virtualKey, bool down)
{
    if (virtualKey >= 256)
        return;
    if (down && !keyDown[virtualKey])
        keyPressed[virtualKey] = true;
    if (!down && keyDown[virtualKey])
        keyReleased[virtualKey] = true;
    keyDown[virtualKey] = down;
}

void Input::OnMouseButton(MouseButton button, bool down)
{
    int i = static_cast<int>(button);
    if (down && !buttonDown[i])
        buttonPressed[i] = true;
    if (!down && buttonDown[i])
        buttonReleased[i] = true;
    buttonDown[i] = down;
}

void Input::OnMouseMove(int dx, int dy)
{
    mouseDx += dx;
    mouseDy += dy;
}

void Input::OnWheel(int notches)
{
    wheel += notches;
}

void Input::NewFrame()
{
    keyPressed.reset();
    keyReleased.reset();
    buttonPressed.reset();
    buttonReleased.reset();
    mouseDx = mouseDy = wheel = 0;
}
