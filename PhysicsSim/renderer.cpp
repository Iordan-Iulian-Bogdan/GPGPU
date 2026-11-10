#include "renderer.h"

#include <d3dcompiler.h>

#include <algorithm>
#include <cstring>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

namespace
{
const char* ShaderSource = R"(
cbuffer Frame : register(b0)
{
    row_major float4x4 viewProjection;
};

struct VSInput
{
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD;
    float4 instancePosition : INSTPOS;
    float4 instanceScale    : INSTSCL;
    float4 instanceColor    : INSTCOL;
    float4 axisX : INSTAX;
    float4 axisY : INSTAY;
    float4 axisZ : INSTAZ;
};

struct VSOutput
{
    float4 position : SV_POSITION;
    float3 normal   : NORMAL;
    float4 color    : COLOR;
};

VSOutput VS(VSInput input)
{
    VSOutput output;
    float3 local = input.position * input.instanceScale.xyz;
    float3 world = local.x * input.axisX.xyz + local.y * input.axisY.xyz + local.z * input.axisZ.xyz
                 + input.instancePosition.xyz;
    output.position = mul(float4(world, 1), viewProjection);
    float3 n = input.normal / input.instanceScale.xyz;
    output.normal = normalize(n.x * input.axisX.xyz + n.y * input.axisY.xyz + n.z * input.axisZ.xyz);
    output.color = input.instanceColor;
    return output;
}

float4 PS(VSOutput input) : SV_Target
{
    float3 n = normalize(input.normal);
    float shade = 0.35 + 0.65 * saturate(dot(n, normalize(float3(-0.4, 0.8, -0.5))));
    return float4(input.color.rgb * shade, 1);
}
)";

constexpr UINT FrameConstantCount = sizeof(Mat4) / 4;
constexpr DXGI_FORMAT BackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr DXGI_FORMAT DepthFormat = DXGI_FORMAT_D32_FLOAT;

D3D12_HEAP_PROPERTIES HeapProperties(D3D12_HEAP_TYPE type)
{
    D3D12_HEAP_PROPERTIES properties = {};
    properties.Type = type;
    return properties;
}

D3D12_RESOURCE_BARRIER TransitionBarrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                         D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    return barrier;
}
}  // namespace

Renderer::~Renderer()
{
    if (queue && fence)
        WaitForGpuIdle();
    if (fenceEvent)
        CloseHandle(fenceEvent);
}

bool Renderer::Init(HWND window, int w, int h)
{
    width = w;
    height = h;

#ifdef _DEBUG
    ComPtr<ID3D12Debug> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
        debug->EnableDebugLayer();
#endif

    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))))
        return false;

    ComPtr<IDXGIAdapter1> adapter;
    factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter));
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))))
        return false;

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue));

    // Tearing support is what lets a windowed swap chain present faster than the monitor refresh.
    BOOL tearing = FALSE;
    factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing));
    swapChainFlags = tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

    DXGI_SWAP_CHAIN_DESC1 swapDesc = {};
    swapDesc.Width = width;
    swapDesc.Height = height;
    swapDesc.Format = BackBufferFormat;
    swapDesc.SampleDesc.Count = 1;
    swapDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapDesc.BufferCount = FrameCount;
    swapDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapDesc.Flags = swapChainFlags;

    ComPtr<IDXGISwapChain1> swapChain1;
    if (FAILED(factory->CreateSwapChainForHwnd(queue.Get(), window, &swapDesc, nullptr, nullptr, &swapChain1)))
        return false;
    factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER);
    swapChain1.As(&swapChain);
    frameIndex = swapChain->GetCurrentBackBufferIndex();

    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc = {};
    rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDesc.NumDescriptors = FrameCount;
    device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&rtvHeap));
    rtvSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_DESCRIPTOR_HEAP_DESC dsvDesc = {};
    dsvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvDesc.NumDescriptors = 1;
    device->CreateDescriptorHeap(&dsvDesc, IID_PPV_ARGS(&dsvHeap));

    CreateRenderTargets();
    CreateDepthBuffer();

    for (UINT i = 0; i < FrameCount; i++)
    {
        device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocators[i]));
        instanceBuffers[i] = CreateUploadBuffer(nullptr, MaxInstances * sizeof(Instance));
        instanceBuffers[i]->Map(0, nullptr, reinterpret_cast<void**>(&instanceData[i]));
    }
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators[0].Get(), nullptr,
                              IID_PPV_ARGS(&commandList));
    commandList->Close();

    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    // Root signature: the view-projection matrix goes through root constants.
    D3D12_ROOT_PARAMETER parameter = {};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameter.Constants.Num32BitValues = FrameConstantCount;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rootDesc = {};
    rootDesc.NumParameters = 1;
    rootDesc.pParameters = &parameter;
    rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    ComPtr<ID3DBlob> rootBlob, errors;
    if (FAILED(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rootBlob, &errors)))
        return false;
    device->CreateRootSignature(0, rootBlob->GetBufferPointer(), rootBlob->GetBufferSize(),
                                IID_PPV_ARGS(&rootSignature));

    UINT compileFlags = 0;
#ifdef _DEBUG
    compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
    ComPtr<ID3DBlob> vs, ps;
    if (FAILED(D3DCompile(ShaderSource, strlen(ShaderSource), nullptr, nullptr, nullptr, "VS", "vs_5_1",
                          compileFlags, 0, &vs, &errors)) ||
        FAILED(D3DCompile(ShaderSource, strlen(ShaderSource), nullptr, nullptr, nullptr, "PS", "ps_5_1",
                          compileFlags, 0, &ps, &errors)))
    {
        if (errors)
            OutputDebugStringA(static_cast<const char*>(errors->GetBufferPointer()));
        return false;
    }

    // Slot 0: mesh vertices. Slot 1: one Instance per object.
    D3D12_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0, offsetof(Vertex, position),   D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT,    0, offsetof(Vertex, normal),     D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       0, offsetof(Vertex, uv),         D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "INSTPOS",  0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, offsetof(Instance, position), D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
        { "INSTSCL",  0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, offsetof(Instance, scale),    D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
        { "INSTCOL",  0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, offsetof(Instance, color),    D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
        { "INSTAX",   0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, offsetof(Instance, axisX),    D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
        { "INSTAY",   0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, offsetof(Instance, axisY),    D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
        { "INSTAZ",   0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, offsetof(Instance, axisZ),    D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1 },
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = rootSignature.Get();
    psoDesc.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
    psoDesc.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
    psoDesc.InputLayout = { layout, _countof(layout) };
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    psoDesc.RasterizerState.DepthClipEnable = TRUE;
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.DepthStencilState.DepthEnable = TRUE;
    psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = BackBufferFormat;
    psoDesc.DSVFormat = DepthFormat;
    psoDesc.SampleDesc.Count = 1;
    return SUCCEEDED(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pipelineState)));
}

void Renderer::CreateRenderTargets()
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < FrameCount; i++)
    {
        swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i]));
        device->CreateRenderTargetView(backBuffers[i].Get(), nullptr, handle);
        handle.ptr += rtvSize;
    }
}

void Renderer::CreateDepthBuffer()
{
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DepthFormat;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    D3D12_CLEAR_VALUE clear = {};
    clear.Format = DepthFormat;
    clear.DepthStencil.Depth = 1.0f;

    D3D12_HEAP_PROPERTIES heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                    &clear, IID_PPV_ARGS(&depthBuffer));
    device->CreateDepthStencilView(depthBuffer.Get(), nullptr, dsvHeap->GetCPUDescriptorHandleForHeapStart());
}

void Renderer::Resize(int w, int h)
{
    if (w <= 0 || h <= 0 || (w == width && h == height) || !swapChain)
        return;

    WaitForGpuIdle();
    for (auto& buffer : backBuffers)
        buffer.Reset();
    depthBuffer.Reset();

    width = w;
    height = h;
    swapChain->ResizeBuffers(FrameCount, width, height, BackBufferFormat, swapChainFlags);
    frameIndex = swapChain->GetCurrentBackBufferIndex();
    CreateRenderTargets();
    CreateDepthBuffer();
}

// data may be null: the buffer is then filled later through its mapped pointer.
ComPtr<ID3D12Resource> Renderer::CreateUploadBuffer(const void* data, UINT64 size)
{
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    D3D12_HEAP_PROPERTIES heap = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
    ComPtr<ID3D12Resource> buffer;
    device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                    nullptr, IID_PPV_ARGS(&buffer));

    if (data)
    {
        void* mapped = nullptr;
        buffer->Map(0, nullptr, &mapped);
        memcpy(mapped, data, size);
        buffer->Unmap(0, nullptr);
    }
    return buffer;
}

GpuMesh Renderer::CreateMesh(const Mesh& mesh)
{
    GpuMesh gpu;

    UINT vertexBytes = static_cast<UINT>(mesh.vertices.size() * sizeof(Vertex));
    gpu.vertexBuffer = CreateUploadBuffer(mesh.vertices.data(), vertexBytes);
    gpu.vertexView = { gpu.vertexBuffer->GetGPUVirtualAddress(), vertexBytes, sizeof(Vertex) };

    UINT indexBytes = static_cast<UINT>(mesh.indices.size() * sizeof(uint32_t));
    gpu.indexBuffer = CreateUploadBuffer(mesh.indices.data(), indexBytes);
    gpu.indexView = { gpu.indexBuffer->GetGPUVirtualAddress(), indexBytes, DXGI_FORMAT_R32_UINT };
    gpu.indexCount = static_cast<uint32_t>(mesh.indices.size());
    return gpu;
}

void Renderer::BeginFrame()
{
    WaitForFrame(frameIndex);
    instancesUsed = 0;

    allocators[frameIndex]->Reset();
    commandList->Reset(allocators[frameIndex].Get(), pipelineState.Get());

    D3D12_RESOURCE_BARRIER toRenderTarget = TransitionBarrier(
        backBuffers[frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    commandList->ResourceBarrier(1, &toRenderTarget);

    D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += static_cast<SIZE_T>(frameIndex) * rtvSize;
    D3D12_CPU_DESCRIPTOR_HANDLE dsv = dsvHeap->GetCPUDescriptorHandleForHeapStart();
    commandList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);

    const float clearColor[] = { 0.08f, 0.09f, 0.12f, 1.0f };
    commandList->ClearRenderTargetView(rtv, clearColor, 0, nullptr);
    commandList->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    D3D12_VIEWPORT viewport = { 0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1 };
    D3D12_RECT scissor = { 0, 0, width, height };
    commandList->RSSetViewports(1, &viewport);
    commandList->RSSetScissorRects(1, &scissor);

    commandList->SetGraphicsRootSignature(rootSignature.Get());
    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
}

void Renderer::DrawInstances(const GpuMesh& mesh, const std::vector<Instance>& instances, const Mat4& viewProjection)
{
    UINT count = static_cast<UINT>(std::min<size_t>(instances.size(), MaxInstances - instancesUsed));
    if (count == 0)
        return;

    memcpy(instanceData[frameIndex] + instancesUsed, instances.data(), count * sizeof(Instance));

    D3D12_VERTEX_BUFFER_VIEW views[2] = {
        mesh.vertexView,
        { instanceBuffers[frameIndex]->GetGPUVirtualAddress() + instancesUsed * sizeof(Instance),
          count * static_cast<UINT>(sizeof(Instance)), sizeof(Instance) },
    };
    commandList->IASetVertexBuffers(0, 2, views);
    commandList->IASetIndexBuffer(&mesh.indexView);
    commandList->SetGraphicsRoot32BitConstants(0, FrameConstantCount, &viewProjection, 0);
    commandList->DrawIndexedInstanced(mesh.indexCount, count, 0, 0, 0);
    instancesUsed += count;
}

void Renderer::EndFrame()
{
    D3D12_RESOURCE_BARRIER toPresent = TransitionBarrier(
        backBuffers[frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    commandList->ResourceBarrier(1, &toPresent);
    commandList->Close();

    ID3D12CommandList* lists[] = { commandList.Get() };
    queue->ExecuteCommandLists(1, lists);
    if (vsync)
        swapChain->Present(1, 0);
    else
        swapChain->Present(0, (swapChainFlags & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING) ? DXGI_PRESENT_ALLOW_TEARING : 0);

    fenceValues[frameIndex] = nextFenceValue;
    queue->Signal(fence.Get(), nextFenceValue++);
    frameIndex = swapChain->GetCurrentBackBufferIndex();
}

void Renderer::WaitForFrame(UINT frame)
{
    if (fence->GetCompletedValue() < fenceValues[frame])
    {
        fence->SetEventOnCompletion(fenceValues[frame], fenceEvent);
        WaitForSingleObject(fenceEvent, INFINITE);
    }
}

void Renderer::WaitForGpuIdle()
{
    UINT64 value = nextFenceValue++;
    queue->Signal(fence.Get(), value);
    fence->SetEventOnCompletion(value, fenceEvent);
    WaitForSingleObject(fenceEvent, INFINITE);
}
