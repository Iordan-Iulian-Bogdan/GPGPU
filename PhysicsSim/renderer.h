#pragma once

#include "instance.h"
#include "mesh.h"

#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <vector>

using Microsoft::WRL::ComPtr;

struct GpuMesh
{
    ComPtr<ID3D12Resource> vertexBuffer;
    ComPtr<ID3D12Resource> indexBuffer;
    D3D12_VERTEX_BUFFER_VIEW vertexView = {};
    D3D12_INDEX_BUFFER_VIEW indexView = {};
    uint32_t indexCount = 0;
};

class Renderer
{
public:
    ~Renderer();

    bool Init(HWND window, int width, int height);
    void Resize(int width, int height);

    GpuMesh CreateMesh(const Mesh& mesh);

    void BeginFrame();
    void DrawInstances(const GpuMesh& mesh, const std::vector<Instance>& instances, const Mat4& viewProjection);
    void EndFrame();

    float AspectRatio() const { return static_cast<float>(width) / static_cast<float>(height); }
    void SetVsync(bool on) { vsync = on; }
    bool Vsync() const { return vsync; }

private:
    static constexpr UINT FrameCount = 2;
    static constexpr UINT MaxInstances = 16384;

    void CreateRenderTargets();
    void CreateDepthBuffer();
    void WaitForGpuIdle();
    void WaitForFrame(UINT frame);
    ComPtr<ID3D12Resource> CreateUploadBuffer(const void* data, UINT64 size);

    int width = 0, height = 0;
    bool vsync = false;
    UINT swapChainFlags = 0;

    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGISwapChain3> swapChain;
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12DescriptorHeap> dsvHeap;
    ComPtr<ID3D12Resource> backBuffers[FrameCount];
    ComPtr<ID3D12Resource> depthBuffer;
    ComPtr<ID3D12CommandAllocator> allocators[FrameCount];
    ComPtr<ID3D12GraphicsCommandList> commandList;
    ComPtr<ID3D12RootSignature> rootSignature;
    ComPtr<ID3D12PipelineState> pipelineState;

    // One instance buffer per frame in flight, filled by DrawInstances.
    ComPtr<ID3D12Resource> instanceBuffers[FrameCount];
    Instance* instanceData[FrameCount] = {};
    UINT instancesUsed = 0;

    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent = nullptr;
    UINT64 fenceValues[FrameCount] = {};
    UINT64 nextFenceValue = 1;

    UINT rtvSize = 0;
    UINT frameIndex = 0;
};
