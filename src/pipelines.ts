import physicsShaderCode from "./shaders/physics.wgsl?raw";
import renderShaderCode from "./shaders/render.wgsl?raw";
import lbvhShaderCode from "./shaders/lbvh.wgsl?raw";
import barnesHutShaderCode from "./shaders/barnes_hut.wgsl?raw";
// @ts-ignore
import { RadixSortKernel } from 'webgpu-radix-sort';
import type { SimBuffers } from "./buffers";
import type { SimConfig } from "./config";


// SIM PIPELINES
export type SimPipelines = {
    physicsShaderModule: GPUShaderModule;
    lbvhShaderModule: GPUShaderModule;
    barnesHutShaderModule: GPUShaderModule;
    halfVelStep: GPUComputePipeline;
    posStep: GPUComputePipeline;
    computeMorton: GPUComputePipeline;
    sortMortonCodes: any;
    buildLBVH: GPUComputePipeline;
    fillLBVH: GPUComputePipeline;
    barnesHutVelStep: GPUComputePipeline;
    barnesHutPosStep: GPUComputePipeline;
};

export function createSimPipelines(device: GPUDevice, config: SimConfig, buffers: SimBuffers): SimPipelines {
    const physicsShaderModule = device.createShaderModule({
        code: physicsShaderCode,
    });
    const lbvhShaderModule = device.createShaderModule({
        code: lbvhShaderCode,
    });
    const barnesHutShaderModule = device.createShaderModule({
        code: barnesHutShaderCode,
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
        values: buffers.bodyIndices,
        count: config.numBodies,
        check_order: false,
        bit_count: 32,
        workgroup_size: { x: 16, y: 16 },
    })

    const buildLBVHPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: lbvhShaderModule,
            entryPoint: "build_lbvh_main",
        },
    });

    const fillLBVHPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: lbvhShaderModule,
            entryPoint: "fill_lbvh_main",
        },
    });

    const barnesHutVelStepPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: barnesHutShaderModule,
            entryPoint: "bh_vel_step_main",
        },
    });

    const barnesHutPosStepPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: barnesHutShaderModule,
            entryPoint: "bh_pos_step_main",
        },
    });

    return {
        physicsShaderModule: physicsShaderModule,
        lbvhShaderModule: lbvhShaderModule,
        barnesHutShaderModule: barnesHutShaderModule,
        halfVelStep: halfVelocityStepPipeline,
        posStep: posStepPipeline,
        computeMorton: computeMortonStepPipeline,
        sortMortonCodes: radixSortKernel,
        buildLBVH: buildLBVHPipeline,
        fillLBVH: fillLBVHPipeline,
        barnesHutVelStep: barnesHutVelStepPipeline,
        barnesHutPosStep: barnesHutPosStepPipeline,
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