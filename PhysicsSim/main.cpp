// Etapa 1 - Prototip 1: 3D Virtual World Physics (CPU), rendered with the MG3D Direct3D 12 renderer.
#include "Physics.h"
#include "camera.h"
#include "importer.h"
#include "input.h"
#include "mesh.h"
#include "renderer.h"

#include <windowsx.h>

#include <cstdio>
#include <filesystem>
#include <vector>

namespace
{
Input input;
Renderer renderer;

void RegisterRawMouse(HWND hwnd)
{
    RAWINPUTDEVICE device = { 0x01, 0x02, 0, hwnd };  // generic desktop, mouse
    RegisterRawInputDevices(&device, 1, sizeof(device));
}

void HandleRawInput(HRAWINPUT handle)
{
    RAWINPUT raw;
    UINT size = sizeof(raw);
    if (GetRawInputData(handle, RID_INPUT, &raw, &size, sizeof(RAWINPUTHEADER)) == static_cast<UINT>(-1))
        return;
    if (raw.header.dwType == RIM_TYPEMOUSE && !(raw.data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE))
        input.OnMouseMove(raw.data.mouse.lLastX, raw.data.mouse.lLastY);
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_SIZE:
        renderer.Resize(LOWORD(lParam), HIWORD(lParam));
        return 0;
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
        input.OnKey(static_cast<UINT>(wParam), true);
        return 0;
    case WM_KEYUP:
    case WM_SYSKEYUP:
        input.OnKey(static_cast<UINT>(wParam), false);
        return 0;
    case WM_LBUTTONDOWN:
        input.OnMouseButton(MouseButton::Left, true);
        return 0;
    case WM_LBUTTONUP:
        input.OnMouseButton(MouseButton::Left, false);
        return 0;
    case WM_MBUTTONDOWN:
        input.OnMouseButton(MouseButton::Middle, true);
        return 0;
    case WM_MBUTTONUP:
        input.OnMouseButton(MouseButton::Middle, false);
        return 0;
    case WM_RBUTTONDOWN:
        input.OnMouseButton(MouseButton::Right, true);
        SetCapture(hwnd);
        ShowCursor(FALSE);
        return 0;
    case WM_RBUTTONUP:
        input.OnMouseButton(MouseButton::Right, false);
        ShowCursor(TRUE);
        ReleaseCapture();
        return 0;
    case WM_MOUSEWHEEL:
        input.OnWheel(GET_WHEEL_DELTA_WPARAM(wParam) / WHEEL_DELTA);
        return 0;
    case WM_INPUT:
        HandleRawInput(reinterpret_cast<HRAWINPUT>(lParam));
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// Loads a .mg3dmesh directly; any other format is imported and cached as .mg3dmesh next to it.
bool LoadModel(const std::filesystem::path& path, Mesh& mesh)
{
    if (path.extension() == ".mg3dmesh")
        return LoadMesh(path.string(), mesh);

    if (!ImportMesh(path.string(), mesh))
        return false;

    std::filesystem::path cached = path;
    cached.replace_extension(".mg3dmesh");
    SaveMesh(mesh, cached.string());
    return true;
}

std::filesystem::path AssetsFolder()
{
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    return std::filesystem::path(exePath).parent_path() / "assets";
}
}  // namespace

int main()
{
    std::printf(
        "Camera: right mouse = look, WASD = move, Space/Ctrl = up/down, Shift = run, wheel = speed\n"
        "  E         spawn 10 mixed objects (sphere/cube/cylinder/cone, random pos + velocity)\n"
        "  K         kick: give random motion to all objects (wakes a settled scene)\n"
        "  C         clear the box\n"
        "  F1/F2/F3  predefined modes: 100/250/500, 250/500/1000, 500/1000/2500 (sphere/cube/cyl+cone)\n"
        "  T         toggle OpenMP on/off (collision pass)\n"
        "  V         toggle VSync on/off\n"
        "  P         pause   |  Esc: quit\n");

    HINSTANCE instance = GetModuleHandleW(nullptr);
    const wchar_t* className = L"PhysicsSimWindow";

    WNDCLASSW wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.lpszClassName = className;
    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(
        0, className, L"Etapa 1 - CPU Physics", WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1280, 720,
        nullptr, nullptr, instance, nullptr);

    RECT client;
    GetClientRect(hwnd, &client);
    if (!renderer.Init(hwnd, client.right, client.bottom))
    {
        MessageBoxW(hwnd, L"Failed to initialize Direct3D 12.", L"PhysicsSim", MB_ICONERROR);
        return 1;
    }
    RegisterRawMouse(hwnd);

    // Same order as the Shape enum: sphere, cube, cylinder, cone.
    const wchar_t* modelFiles[4] = { L"sphere.obj", L"cube.obj", L"cylinder.obj", L"cone.obj" };
    GpuMesh meshes[4];
    for (int i = 0; i < 4; i++)
    {
        std::filesystem::path path = AssetsFolder() / modelFiles[i];
        Mesh mesh;
        if (!LoadModel(path, mesh))
        {
            MessageBoxW(hwnd, (L"Failed to load model: " + path.wstring()).c_str(), L"PhysicsSim", MB_ICONERROR);
            return 1;
        }
        meshes[i] = renderer.CreateMesh(mesh);
    }

    ShowWindow(hwnd, SW_SHOW);

    World world;
    world.LoadPreset(0);
    std::vector<Instance> instances[4];
    bool paused = false;

    Camera camera;
    camera.position = { 0, 10, -34 };
    camera.moveSpeed = 10.0f;

    LARGE_INTEGER frequency, last;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&last);
    float fpsTimer = 0;
    int fpsFrames = 0;
    double physicsMs = 0;

    MSG msg = {};
    while (msg.message != WM_QUIT)
    {
        if (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
            continue;
        }

        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        float dt = static_cast<float>(now.QuadPart - last.QuadPart) / static_cast<float>(frequency.QuadPart);
        last = now;

        if (input.WasPressed(VK_ESCAPE))
            DestroyWindow(hwnd);
        if (input.WasPressed('E'))
            for (int i = 0; i < 10; i++)
                world.Spawn(static_cast<Shape>(i % 4), true);
        if (input.WasPressed('K')) world.Kick();
        if (input.WasPressed('C')) world.Clear();
        if (input.WasPressed('P')) paused = !paused;
        if (input.WasPressed('T')) world.SetParallel(!world.Parallel());
        if (input.WasPressed('V')) renderer.SetVsync(!renderer.Vsync());
        for (UINT key = VK_F1; key <= VK_F3; key++)
            if (input.WasPressed(key)) world.LoadPreset(key - VK_F1);

        camera.Update(input, dt);

        if (!paused)
        {
            LARGE_INTEGER start, end;
            QueryPerformanceCounter(&start);
            world.Step(1.0f / 60.0f);
            QueryPerformanceCounter(&end);
            physicsMs += 1000.0 * (end.QuadPart - start.QuadPart) / frequency.QuadPart;
        }

        world.BuildInstances(instances);
        Mat4 projection = Perspective(ToRadians(60.0f), renderer.AspectRatio(), 0.1f, 300.0f);
        Mat4 viewProjection = camera.ViewMatrix() * projection;

        renderer.BeginFrame();
        for (int i = 0; i < 4; i++)
            renderer.DrawInstances(meshes[i], instances[i], viewProjection);
        renderer.EndFrame();

        input.NewFrame();

        fpsTimer += dt;
        fpsFrames++;
        if (fpsTimer >= 0.5f)
        {
            wchar_t title[256];
            swprintf_s(title, L"Etapa 1 - CPU Physics (OpenMP %s, VSync %s) | objects: %zu | FPS: %.1f | physics: %.2f ms/frame%s",
                       world.Parallel() ? L"ON" : L"OFF", renderer.Vsync() ? L"ON" : L"OFF", world.Bodies().size(),
                       fpsFrames / fpsTimer, physicsMs / fpsFrames, paused ? L" | PAUSED" : L"");
            SetWindowTextW(hwnd, title);
            fpsTimer = 0;
            fpsFrames = 0;
            physicsMs = 0;
        }
    }
    return 0;
}
