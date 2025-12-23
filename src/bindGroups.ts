import type { RenderBuffers, SimBuffers } from "./buffers";
import type { RenderPipelines, SimPipelines } from "./pipelines";


// SIMULATION BIND GROUPS
export type SimBindGroups = Readonly<{
    halfVelStep: GPUBindGroup;
    posStep: GPUBindGroup;
    computeMortonStep: GPUBindGroup;
    buildLBVHStep: GPUBindGroup;
    fillLBVHStep: GPUBindGroup;
}>;

export function createSimBindGroups(device: GPUDevice, buffers: SimBuffers, pipelines: SimPipelines): SimBindGroups {
    const halfVelStepBindGroup = device.createBindGroup({
        layout: pipelines.halfVelStep.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: buffers.floatMetadata } },
            { binding: 1, resource: { buffer: buffers.uintMetadata } },
            { binding: 2, resource: { buffer: buffers.mass } },
            { binding: 3, resource: { buffer: buffers.pos } },
            { binding: 4, resource: { buffer: buffers.vel } },
        ]
    });

    const posStepBindGroup = device.createBindGroup({
        layout: pipelines.posStep.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: buffers.floatMetadata } },
            { binding: 1, resource: { buffer: buffers.uintMetadata } },
            // { binding: 2, resource: { buffer: buffers.mass } },
            { binding: 3, resource: { buffer: buffers.pos } },
            { binding: 4, resource: { buffer: buffers.vel } },
        ]
    });

    const computeMortonStepBindGroup = device.createBindGroup({
        layout: pipelines.computeMortonStep.getBindGroupLayout(0),
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

    return {
        halfVelStep: halfVelStepBindGroup,
        posStep: posStepBindGroup,
        computeMortonStep: computeMortonStepBindGroup,
        buildLBVHStep: buildLBVHBindGroup,
        fillLBVHStep: fillLBVHBindGroup,
    };
}


// RENDERING BIND GROUPS
export type RenderBindGroups = Readonly<{
    render: GPUBindGroup;
}>;

export function createRenderBindGroups(device: GPUDevice, buffers: RenderBuffers, pipelines: RenderPipelines): RenderBindGroups {
    const renderBindGroup = device.createBindGroup({
        layout: pipelines.render.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: buffers.floatMetadata } },
            { binding: 1, resource: { buffer: buffers.uintMetadata } },
            { binding: 2, resource: { buffer: buffers.pos } },
        ]
    });

    return {
        render: renderBindGroup,
    };
}