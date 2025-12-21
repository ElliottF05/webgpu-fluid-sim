import type { SimConfig } from "./config";


// HELPER FUNCTIONS
function createBuffer(device: GPUDevice, initialData: Uint32Array | Float32Array): GPUBuffer {
    const buffer = device.createBuffer({
        size: initialData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, initialData.buffer, initialData.byteOffset, initialData.byteLength);
    return buffer;
}
function createMetadataBuffer(device: GPUDevice, initialData: Uint32Array | Float32Array): GPUBuffer {
    const buffer = device.createBuffer({
        size: initialData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, initialData.buffer, initialData.byteOffset, initialData.byteLength);
    return buffer;
}


// SIMULATION BUFFERS
export type SimBuffers = {
    floatMetadata: GPUBuffer;
    uintMetadata: GPUBuffer;
    mass: GPUBuffer;
    pos: GPUBuffer;
    vel: GPUBuffer;
};

export function createSimBuffers(device: GPUDevice, config: SimConfig, camCenter: [number, number], camHalfSize: [number, number], viewPort: [number, number]): SimBuffers {
    // metadata buffers
    const { uintMetadataArray, floatMetadataArray } = buildMetadataArrays(config, camCenter, camHalfSize, viewPort);
    const uintMetadataBuffer = createMetadataBuffer(device, uintMetadataArray);
    const floatMetadataBuffer = createMetadataBuffer(device,floatMetadataArray);

    // mass buffer
    const massArray = new Float32Array(config.numBodies).fill(1.0);
    massArray[1] = 10000.0;
    const massBuffer = createBuffer(device, massArray);

    // pos buffer
    const initPosArray = new Float32Array(2 * config.numBodies).fill(0.0);
    for (let i = 0; i < config.numBodies; i++) {
        const x = (Math.random() - 0.5) * 15;
        const y = (Math.random() - 0.5) * 15;
        initPosArray[2 * i] = x;
        initPosArray[2 * i + 1] = y;
    }
    initPosArray[2] = 20.0;
    const posBuffer = createBuffer(device, initPosArray);

    // velocity buffer
    const initVelArray = new Float32Array(2 * config.numBodies).fill(0.0);
    for (let i = 0; i < config.numBodies; i++) {
        const x = (Math.random() - 0.5) * 1.0;
        const y = (Math.random() - 0.5) * 1.0;
        initVelArray[2 * i] = x;
        initVelArray[2 * i + 1] = y;
    }
    initVelArray[3] = 20.0;
    const velBuffer = createBuffer(device, initVelArray);

    return {
        floatMetadata: floatMetadataBuffer,
        uintMetadata: uintMetadataBuffer,
        mass: massBuffer,
        pos: posBuffer,
        vel: velBuffer,
    };
}

export function buildMetadataArrays(config: SimConfig, camCenter: [number, number], camHalfSize: [number, number], viewPort: [number, number]): { uintMetadataArray: Uint32Array; floatMetadataArray: Float32Array } {
    const uintMetadataArray = new Uint32Array([config.numBodies]);
    const floatMetadataArray = new Float32Array([config.gravConstant, config.deltaTime, config.epsilonMultiplier, 0.0, ...camCenter, ...camHalfSize, ...viewPort]);
    return { uintMetadataArray, floatMetadataArray };
}


// RENDERING BUFFERS
export type RenderBuffers = {
    floatMetadata: GPUBuffer;
    uintMetadata: GPUBuffer;
    pos: GPUBuffer;
};

export function createRenderBuffers(simBuffers: SimBuffers): RenderBuffers {
    return {
        floatMetadata: simBuffers.floatMetadata,
        uintMetadata: simBuffers.uintMetadata,
        pos: simBuffers.pos,
    };
}