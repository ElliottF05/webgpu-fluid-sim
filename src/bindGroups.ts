import type { RenderBuffers, SimBuffers } from "./buffers";
import type { RenderPipelines, SimPipelines } from "./pipelines";


// SIMULATION BIND GROUPS
export type SimBindGroups = Readonly<{
    computeMortonStep: GPUBindGroup;
    buildLBVHStep: GPUBindGroup;
    fillLBVHStep: GPUBindGroup;
    barnesHutVelStep: GPUBindGroup;
    barnesHutPosStep: GPUBindGroup;
}>;

export function createSimBindGroups(device: GPUDevice, buffers: SimBuffers, pipelines: SimPipelines): SimBindGroups {
    const computeMortonStepBindGroup = device.createBindGroup({
        layout: pipelines.computeMorton.getBindGroupLayout(0),
        entries: [
            { binding: 1, resource: { buffer: buffers.uintMetadata } },
            { binding: 2, resource: { buffer: buffers.pos } },
            { binding: 4, resource: { buffer: buffers.mortonCodes } },
            { binding: 5, resource: { buffer: buffers.bodyIndices } },
        ]
    });

    const buildLBVHBindGroup = device.createBindGroup({
        layout: pipelines.buildLBVH.getBindGroupLayout(0),
        entries: [
            { binding: 1, resource: { buffer: buffers.uintMetadata } },
            { binding: 4, resource: { buffer: buffers.mortonCodes } },
            { binding: 6, resource: { buffer: buffers.nodeData } },
            { binding: 7, resource: { buffer: buffers.nodeStatus } },
        ]
    });

    const fillLBVHBindGroup = device.createBindGroup({
        layout: pipelines.fillLBVH.getBindGroupLayout(0),
        entries: [
            { binding: 1, resource: { buffer: buffers.uintMetadata } },
            { binding: 2, resource: { buffer: buffers.pos } },
            { binding: 3, resource: { buffer: buffers.mass } },
            { binding: 5, resource: { buffer: buffers.bodyIndices } },
            { binding: 6, resource: { buffer: buffers.nodeData } },
            { binding: 7, resource: { buffer: buffers.nodeStatus } },
        ]
    });

    const barnesHutVelStepBindGroup = device.createBindGroup({
        layout: pipelines.barnesHutVelStep.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: buffers.floatMetadata } },
            { binding: 1, resource: { buffer: buffers.uintMetadata } },
            { binding: 2, resource: { buffer: buffers.pos } },
            { binding: 3, resource: { buffer: buffers.vel } },
            { binding: 4, resource: { buffer: buffers.mass } },
            { binding: 5, resource: { buffer: buffers.bodyIndices } },
            { binding: 6, resource: { buffer: buffers.nodeData } },
        ]
    });

    const barnesHutPosStepBindGroup = device.createBindGroup({
        layout: pipelines.barnesHutPosStep.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: buffers.floatMetadata } },
            { binding: 1, resource: { buffer: buffers.uintMetadata } },
            { binding: 2, resource: { buffer: buffers.pos } },
            { binding: 3, resource: { buffer: buffers.vel } },
            { binding: 5, resource: { buffer: buffers.bodyIndices } },
        ]
    });

    return {
        computeMortonStep: computeMortonStepBindGroup,
        buildLBVHStep: buildLBVHBindGroup,
        fillLBVHStep: fillLBVHBindGroup,
        barnesHutVelStep: barnesHutVelStepBindGroup,
        barnesHutPosStep: barnesHutPosStepBindGroup,
    };
}


// RENDERING BIND GROUPS
export type RenderBindGroups = Readonly<{
    density: GPUBindGroup;
    toneMap: GPUBindGroup;
}>;

export function createRenderBindGroups(device: GPUDevice, buffers: RenderBuffers, pipelines: RenderPipelines): RenderBindGroups {
    // adjust bindings as needed
    const densityBindGroup = device.createBindGroup({
        layout: pipelines.density.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: buffers.floatMetadata },
            { binding: 1, resource: buffers.uintMetadata },
            { binding: 2, resource: buffers.pos },
            { binding: 3, resource: buffers.densityTextureView },
        ],
    });

    // adjust these as needed too
    const toneMapBindGroup = device.createBindGroup({
        layout: pipelines.toneMap.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: buffers.floatMetadata },
            { binding: 1, resource: buffers.uintMetadata },
            { binding: 3, resource: buffers.densityTextureView },
        ],
    });

    return {
        density: densityBindGroup,
        toneMap: toneMapBindGroup,
    };
}