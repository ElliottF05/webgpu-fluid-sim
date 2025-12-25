import type { SimConfig } from "./config";


// HELPER FUNCTIONS
function createBuffer(device: GPUDevice, initialData: Uint32Array | Float32Array): GPUBuffer {
    const buffer = device.createBuffer({
        size: initialData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
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

// DEBUG: read buffer back to CPU
export async function readBufferData<T extends Uint32Array | Float32Array>(device: GPUDevice, buffer: GPUBuffer, size: number, ArrayType: { new(buffer: ArrayBuffer): T; BYTES_PER_ELEMENT: number; }): Promise<T> {
    const readBuffer = device.createBuffer({
        size: size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size);
    const commands = commandEncoder.finish();
    device.queue.submit([commands]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = readBuffer.getMappedRange();
    const data = new ArrayType(arrayBuffer.slice(0));
    readBuffer.unmap();
    return data;
}


// SIMULATION BUFFERS
export type SimBuffers = {
    floatMetadata: GPUBuffer;
    uintMetadata: GPUBuffer;
    mass: GPUBuffer;
    pos: GPUBuffer;
    vel: GPUBuffer;
    mortonCodes: GPUBuffer;
    bodyIndices: GPUBuffer;
    nodeData: GPUBuffer;
    nodeStatus: GPUBuffer;
};

export function createSimBuffers(device: GPUDevice, config: SimConfig, numBodies: number, camCenter: [number, number], camHalfSize: [number, number], viewPort: [number, number]): SimBuffers {
    // metadata buffers
    const { uintMetadataArray, floatMetadataArray } = buildMetadataArrays(config, numBodies, camCenter, camHalfSize, viewPort);
    const uintMetadataBuffer = createMetadataBuffer(device, uintMetadataArray);
    const floatMetadataBuffer = createMetadataBuffer(device,floatMetadataArray);

    // mass buffer
    const massArray = new Float32Array(numBodies).fill(1.0);
    // massArray[1] = 10000.0;
    const massBuffer = createBuffer(device, massArray);

    // pos buffer
    const initPosArray = new Float32Array(2 * numBodies).fill(0.0);
    for (let i = 0; i < numBodies; i++) {
        // distribute using 2d gaussian
        const angle = Math.random() * 2.0 * Math.PI;
        const radius = Math.sqrt(-2.0 * Math.log(Math.random())) * 5.0;
        const x = radius * Math.cos(angle);
        const y = radius * Math.sin(angle);
        initPosArray[2 * i] = x;
        initPosArray[2 * i + 1] = y;
    }
    initPosArray[2] = 20.0;
    const posBuffer = createBuffer(device, initPosArray);

    // velocity buffer
    const initVelArray = new Float32Array(2 * numBodies).fill(0.0);
    for (let i = 0; i < numBodies; i++) {
        const x = initPosArray[2 * i];
        const y = initPosArray[2 * i + 1];
        // circular velocity
        const dist = Math.sqrt(x * x + y * y) + 0.1;
        const speed = 10 * Math.sqrt(100.0 / dist);
        const angle = Math.atan2(y, x) + Math.PI / 2.0;
        const vx = speed * Math.cos(angle);
        const vy = speed * Math.sin(angle);
        initVelArray[2 * i] = vx;
        initVelArray[2 * i + 1] = vy;
    }
    // initVelArray[3] = 20.0;
    const velBuffer = createBuffer(device, initVelArray);

    // morton codes buffer
    const mortonCodesArray = new Uint32Array(numBodies).fill(0);
    const mortonCodesBuffer = createBuffer(device, mortonCodesArray);

    // indices buffer
    const bodyIndicesArray = new Uint32Array(numBodies).fill(0);
    const bodyIndicesBuffer = createBuffer(device, bodyIndicesArray);

    // node data buffer (each node_data is 12 * 4 = 48 bytes):
    // struct NodeData {
    //     center_of_mass: vec2<f32>,
    //     aabb_min: vec2<f32>,
    //     aabb_max: vec2<f32>,
    //     total_mass: f32,
    //     length: f32,
    //     left_child: u32,
    //     right_child: u32,
    //     parent: u32,
    //     _pad: u32,
    // }
    const numNodes = 2 * numBodies - 1;
    const nodeDataArray = new Uint32Array(numNodes * 12).fill(0.0);
    const nodeDataBuffer = createBuffer(device, nodeDataArray);

    const nodeStatusArray = new Uint32Array(numNodes).fill(0);
    const nodeStatusBuffer = createBuffer(device, nodeStatusArray);

    return {
        floatMetadata: floatMetadataBuffer,
        uintMetadata: uintMetadataBuffer,
        mass: massBuffer,
        pos: posBuffer,
        vel: velBuffer,
        mortonCodes: mortonCodesBuffer,
        bodyIndices: bodyIndicesBuffer,
        nodeData: nodeDataBuffer,
        nodeStatus: nodeStatusBuffer,
    };
}

export function buildMetadataArrays(config: SimConfig, numBodies: number, camCenter: [number, number], camHalfSize: [number, number], viewPort: [number, number]): { uintMetadataArray: Uint32Array; floatMetadataArray: Float32Array } {
    const uintMetadataArray = new Uint32Array([numBodies]);
    const floatMetadataArray = new Float32Array([config.gravConstant, config.deltaTime, config.epsilonMultiplier, config.bhTheta, ...camCenter, ...camHalfSize, ...viewPort]);
    return { uintMetadataArray, floatMetadataArray };
}


// RENDERING BUFFERS
export type RenderBuffers = {
    floatMetadata: GPUBuffer;
    uintMetadata: GPUBuffer;
    pos: GPUBuffer;
    densityTexture: GPUTexture;
    densityTextureView: GPUTextureView;
    densityTextureSampler: GPUSampler;
};

export function createRenderBuffers(device: GPUDevice, simBuffers: SimBuffers, viewPort: [number, number]): RenderBuffers {
    const densityTexture = device.createTexture({
        size: [viewPort[0], viewPort[1]],
        format: "r16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const densityTextureSampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
    });

    return {
        floatMetadata: simBuffers.floatMetadata,
        uintMetadata: simBuffers.uintMetadata,
        pos: simBuffers.pos,
        densityTexture: densityTexture,
        densityTextureView: densityTexture.createView(),
        densityTextureSampler: densityTextureSampler,
    }
}