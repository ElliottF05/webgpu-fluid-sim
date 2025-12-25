import lbvhShaderCode from "./shaders/compute/lbvh.wgsl?raw";
import barnesHutShaderCode from "./shaders/compute/barnes_hut.wgsl?raw";
import densityShaderCode from "./shaders/render/density.wgsl?raw";
import toneMapShaderCode from "./shaders/render/tonemap.wgsl?raw";
// @ts-ignore
import { RadixSortKernel } from 'webgpu-radix-sort';
import type { Simulation } from "./simulation";


// SIM PIPELINES
export type SimPipelines = {
    lbvhShaderModule: GPUShaderModule;
    barnesHutShaderModule: GPUShaderModule;
    computeMorton: GPUComputePipeline;
    sortMortonCodes: any;
    buildLBVH: GPUComputePipeline;
    fillLBVH: GPUComputePipeline;
    barnesHutVelStep: GPUComputePipeline;
    barnesHutPosStep: GPUComputePipeline;
};

export function createSimPipelines(device: GPUDevice, sim: Simulation): SimPipelines {
    const lbvhShaderModule = device.createShaderModule({
        code: lbvhShaderCode,
    });
    const barnesHutShaderModule = device.createShaderModule({
        code: barnesHutShaderCode,
    });;

    const computeMortonStepPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: lbvhShaderModule,
            entryPoint: "compute_morton_codes_main",
        },
    });

    const radixSortKernel = new RadixSortKernel({
        device: device,
        keys: sim.getBuffers().mortonCodes,
        values: sim.getBuffers().bodyIndices,
        count: sim.getNumBodies(),
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
        lbvhShaderModule: lbvhShaderModule,
        barnesHutShaderModule: barnesHutShaderModule,
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
    densityShaderModule: GPUShaderModule;
    toneMapShaderModule: GPUShaderModule;
    density: GPURenderPipeline;
    toneMap: GPURenderPipeline;
};

export function createRenderPipelines(device: GPUDevice, canvasFormat: GPUTextureFormat): RenderPipelines {
    const densityShaderModule = device.createShaderModule({
        code: densityShaderCode,
    });
    const toneMapShaderModule = device.createShaderModule({
        code: toneMapShaderCode,
    });

    const densityPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: densityShaderModule,
            entryPoint: "vertex_main",
        },
        fragment: {
            module: densityShaderModule,
            entryPoint: "fragment_main",
            targets: [{ 
                format: "r16float",
                writeMask: GPUColorWrite.ALL,
            }],
        },
        primitive: {
            topology: "triangle-list",
        },
    });

    const toneMapPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: toneMapShaderModule,
            entryPoint: "vertex_main",
        },
        fragment: {
            module: toneMapShaderModule,
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
        densityShaderModule: densityShaderModule,
        toneMapShaderModule: toneMapShaderModule,
        density: densityPipeline,
        toneMap: toneMapPipeline,
    };
}