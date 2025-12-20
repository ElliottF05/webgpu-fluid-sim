import physicsShaderCode from "./shaders/physics.wgsl?raw";
import renderShaderCode from "./shaders/render.wgsl?raw";


// SIM PIPELINES
export type SimPipelines = {
    shaderModule: GPUShaderModule;
    halfVelStep: GPUComputePipeline;
    posStep: GPUComputePipeline;
};

export function createSimPipelines(device: GPUDevice): SimPipelines {
    const shaderModule = device.createShaderModule({
        code: physicsShaderCode,
    });

    const halfVelocityStepPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: shaderModule,
            entryPoint: "half_vel_step_main",
        },
    });

    const posStepPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: shaderModule,
            entryPoint: "pos_step_main",
        },
    });

    return {
        shaderModule,
        halfVelStep: halfVelocityStepPipeline,
        posStep: posStepPipeline,
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