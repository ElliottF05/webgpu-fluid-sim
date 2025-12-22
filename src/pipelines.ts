import physicsShaderCode from "./shaders/physics.wgsl?raw";
import renderShaderCode from "./shaders/render.wgsl?raw";
import lbvhShaderCode from "./shaders/lbvh.wgsl?raw";
// @ts-ignore
import { RadixSortKernel } from 'webgpu-radix-sort';
import type { SimBuffers } from "./buffers";


// SIM PIPELINES
export type SimPipelines = {
    physicsShaderModule: GPUShaderModule;
    lbvhShaderModule: GPUShaderModule;
    halfVelStep: GPUComputePipeline;
    posStep: GPUComputePipeline;
    computeMortonStep: GPUComputePipeline;
    sortMortonCodes: any;
};

export function createSimPipelines(device: GPUDevice, buffers: SimBuffers): SimPipelines {
    const physicsShaderModule = device.createShaderModule({
        code: physicsShaderCode,
    });
    const lbvhShaderModule = device.createShaderModule({
        code: lbvhShaderCode,
    });

    const halfVelocityStepPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: physicsShaderModule,
            entryPoint: "half_vel_step_main",
        },
    });

    const posStepPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: physicsShaderModule,
            entryPoint: "pos_step_main",
        },
    });

    const computeMortonStepPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: lbvhShaderModule,
            entryPoint: "compute_morton_codes_main",
        },
    });

    const radixSortKernel = new RadixSortKernel({
        device: device,
        keys: buffers.mortonCodes,
        values: buffers.indices,
        count: buffers.mortonCodes.size / 4,
        check_order: false,
        bit_count: 32,
        workgroup_size: { x: 16, y: 16 },
    })

    return {
        physicsShaderModule: physicsShaderModule,
        lbvhShaderModule: lbvhShaderModule,
        halfVelStep: halfVelocityStepPipeline,
        posStep: posStepPipeline,
        computeMortonStep: computeMortonStepPipeline,
        sortMortonCodes: radixSortKernel,
    };
}


// RENDER PIPELINES
export type RenderPipelines = {
    shaderModule: GPUShaderModule;
    render: GPURenderPipeline;
};

export function createRenderPipelines(device: GPUDevice, canvasFormat: GPUTextureFormat): RenderPipelines {
    const shaderModule = device.createShaderModule({
        code: renderShaderCode,
    });

    const renderPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: shaderModule,
            entryPoint: "vertex_main",
        },
        fragment: {
            module: shaderModule,
            entryPoint: "fragment_main",
            targets: [{ 
            format: canvasFormat,
            blend: {
                color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
                alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add", },
            },
            writeMask: GPUColorWrite.ALL,
            }],
        },
        primitive: {
            topology: "triangle-list",
        },
    });

    return {
        shaderModule,
        render: renderPipeline,
    };
}