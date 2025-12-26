import type { GPUCommandSource } from "./main";
import { type Config } from "./config";
import lbvhShaderCode from "./shaders/compute/lbvh.wgsl?raw";
import barnesHutShaderCode from "./shaders/compute/barnes_hut.wgsl?raw";

// @ts-ignore
import { RadixSortKernel } from "webgpu-radix-sort";


type SimBuffers = {
    metadata: GPUBuffer;
    mass: GPUBuffer;
    pos: GPUBuffer;
    vel: GPUBuffer;
    mortonCodes: GPUBuffer;
    bodyIndices: GPUBuffer;
    nodeData: GPUBuffer;
    nodeStatus: GPUBuffer;
};

type SimPipelines = {
    computeMorton: GPUComputePipeline;
    sortMorton: any; // no ts type exposed for radix sort library
    buildLBVH: GPUComputePipeline;
    fillLBVH: GPUComputePipeline;
    barnesHutVelStep: GPUComputePipeline;
    barnesHutPosStep: GPUComputePipeline;
};

type SimBindGroups = {
    computeMorton: GPUBindGroup;
    buildLBVH: GPUBindGroup;
    fillLBVH: GPUBindGroup;
    barnesHutVelStep: GPUBindGroup;
    barnesHutPosStep: GPUBindGroup;
};

type SimScenario = "default" | "others";

export class Simulation implements GPUCommandSource {
    // immutable config
    private readonly config: Config;

    // gpu device
    private readonly device: GPUDevice;

    // current scenario
    private currentScenario: SimScenario;

    // num bodies
    private numBodies: number;

    // GPU buffers, pipelines, and bind groups
    private buffers: SimBuffers;
    private pipelines: SimPipelines;
    private bindGroups: SimBindGroups;


    // INITIALIATION

    public constructor(config: Config, device: GPUDevice) {
        this.config = config;
        this.device = device;

        this.currentScenario = "default";

        // set up num bodies
        this.numBodies = 50000;

        // set up GPU buffers, pipelines, and bind groups
        this.buffers = this.createSimBuffers();
        this.pipelines = this.createSimPipelines();
        this.bindGroups = this.createSimBindGroups();

        // fill buffers with initial data
        this.updateMetadataBuffer();
        this.setScenario(this.currentScenario);
    }

    private createSimBuffers(): SimBuffers {
        const metadata = this.device.createBuffer({
            size: 8 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // helper to create storage buffers
        const createStorageBuffer = (byteSize: number): GPUBuffer => {
            return this.device.createBuffer({
                size: byteSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
        }

        const mass = createStorageBuffer(this.numBodies * 4);
        const pos = createStorageBuffer(this.numBodies * 2 * 4);
        const vel = createStorageBuffer(this.numBodies * 2 * 4);
        const mortonCodes = createStorageBuffer(this.numBodies * 4);
        const bodyIndices = createStorageBuffer(this.numBodies * 4);

        const numNodes = 2 * this.numBodies - 1;
        const nodeData = createStorageBuffer(numNodes * 12 * 4);
        const nodeStatus = createStorageBuffer(numNodes * 4);

        return {
            metadata,
            mass,
            pos,
            vel,
            mortonCodes,
            bodyIndices,
            nodeData,
            nodeStatus,
        };
    }

    private createSimPipelines(): SimPipelines {
        const lbvhShaderModule = this.device.createShaderModule({
            code: lbvhShaderCode,
        });
        const barnesHutShaderModule = this.device.createShaderModule({
            code: barnesHutShaderCode,
        });

        // helper to create compute pipelines
        const createComputePipeline = (module: GPUShaderModule, entryPoint: string): GPUComputePipeline => {
            return this.device.createComputePipeline({
                layout: "auto",
                compute: {
                    module: module,
                    entryPoint: entryPoint,
                },
            });
        }

        const computeMorton = createComputePipeline(lbvhShaderModule, "compute_morton_codes_main");
        const buildLBVH = createComputePipeline(lbvhShaderModule, "build_lbvh_main");
        const fillLBVH = createComputePipeline(lbvhShaderModule, "fill_lbvh_main");
        const barnesHutVelStep = createComputePipeline(barnesHutShaderModule, "bh_vel_step_main");
        const barnesHutPosStep = createComputePipeline(barnesHutShaderModule, "bh_pos_step_main");

        // use external radix sort library for sorting morton codes
        const sortMorton = new RadixSortKernel({
            device: this.device,
            keys: this.buffers.mortonCodes,
            values: this.buffers.bodyIndices,
            count: this.numBodies,
            check_order: false,
            bit_count: 32,
            workgroup_size: { x: 16, y: 16 },
        });

        return {
            computeMorton,
            sortMorton,
            buildLBVH,
            fillLBVH,
            barnesHutVelStep,
            barnesHutPosStep,
        };
    }

    private createSimBindGroups(): SimBindGroups {
        const computeMorton = this.device.createBindGroup({
            layout: this.pipelines.computeMorton.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.metadata } },
                { binding: 2, resource: { buffer: this.buffers.pos } },
                { binding: 4, resource: { buffer: this.buffers.mortonCodes } },
                { binding: 5, resource: { buffer: this.buffers.bodyIndices } },
            ]
        });
        const buildLBVH = this.device.createBindGroup({
            layout: this.pipelines.buildLBVH.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.metadata } },
                { binding: 4, resource: { buffer: this.buffers.mortonCodes } },
                { binding: 6, resource: { buffer: this.buffers.nodeData } },
                { binding: 7, resource: { buffer: this.buffers.nodeStatus } },
            ]
        });
        const fillLBVH = this.device.createBindGroup({
            layout: this.pipelines.fillLBVH.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.metadata } },
                { binding: 1, resource: { buffer: this.buffers.mass } },
                { binding: 2, resource: { buffer: this.buffers.pos } },
                { binding: 5, resource: { buffer: this.buffers.bodyIndices } },
                { binding: 6, resource: { buffer: this.buffers.nodeData } },
                { binding: 7, resource: { buffer: this.buffers.nodeStatus } },
            ]
        });
        const barnesHutVelStep = this.device.createBindGroup({
            layout: this.pipelines.barnesHutVelStep.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.metadata } },
                { binding: 1, resource: { buffer: this.buffers.mass } },
                { binding: 2, resource: { buffer: this.buffers.pos } },
                { binding: 3, resource: { buffer: this.buffers.vel } },
                { binding: 5, resource: { buffer: this.buffers.bodyIndices } },
                { binding: 6, resource: { buffer: this.buffers.nodeData } },
            ]
        });
        const barnesHutPosStep = this.device.createBindGroup({
            layout: this.pipelines.barnesHutPosStep.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.metadata } },
                { binding: 2, resource: { buffer: this.buffers.pos } },
                { binding: 3, resource: { buffer: this.buffers.vel } },
                { binding: 5, resource: { buffer: this.buffers.bodyIndices } },
            ]
        });

        return {
            computeMorton,
            buildLBVH,
            fillLBVH,
            barnesHutVelStep,
            barnesHutPosStep,
        };
    }

    public updateMetadataBuffer() {
        const metadataArray = new ArrayBuffer(8 * 4);
        const metadataFloatView = new Float32Array(metadataArray);
        const metadataUintView = new Uint32Array(metadataArray);

        metadataUintView[0] = this.numBodies;
        metadataFloatView[1] = this.config.gravConstant;
        metadataFloatView[2] = this.config.deltaTime;
        metadataFloatView[3] = this.config.epsilonMultiplier;
        metadataFloatView[4] = this.config.bhTheta;

        this.device.queue.writeBuffer(this.buffers.metadata, 0, metadataArray);
    }

    public setNumBodies(numBodies: number) {
        this.numBodies = numBodies;
        this.updateMetadataBuffer();
        this.setScenario(this.currentScenario);
    }

    public setNumBodiesAndScenario(numBodies: number, scenario: SimScenario) {
        this.numBodies = numBodies;
        this.updateMetadataBuffer();
        this.setScenario(scenario);
        this.currentScenario = scenario;
    }

    public setScenario(scenario: SimScenario) {
        // for now, only default scenario
        this.currentScenario = scenario;
        if (scenario === "default") {
            const massData = new Float32Array(this.numBodies);
            const posData = new Float32Array(this.numBodies * 2);
            const velData = new Float32Array(this.numBodies * 2);

            // simple random distribution
            for (let i = 0; i < this.numBodies; i++) {
                massData[i] = Math.random() * 5.0 + 1.0; // mass between 1 and 6

                posData[2 * i] = (Math.random() - 0.5) * 20.0; // x position
                posData[2 * i + 1] = (Math.random() - 0.5) * 20.0; // y position

                velData[2 * i] = (Math.random() - 0.5) * 1.0; // x velocity
                velData[2 * i + 1] = (Math.random() - 0.5) * 1.0; // y velocity
            }
            this.updatePhysicsBuffers(massData, posData, velData);
        }
    }

    private updatePhysicsBuffers(massData: Float32Array, posData: Float32Array, velData: Float32Array) {
        this.device.queue.writeBuffer(this.buffers.mass, 0, massData.buffer, massData.byteOffset, massData.byteLength);
        this.device.queue.writeBuffer(this.buffers.pos, 0, posData.buffer, posData.byteOffset, posData.byteLength);
        this.device.queue.writeBuffer(this.buffers.vel, 0, velData.buffer, velData.byteOffset, velData.byteLength);
    }

    public getCommands(): GPUCommandBuffer {
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();

        const workgroupSize = 64;
        const dispatchCount = Math.ceil(this.numBodies / workgroupSize);

        for (let step = 0; step < this.config.substeps; step++) {
            // compute morton codes step
            computePass.setPipeline(this.pipelines.computeMorton);
            computePass.setBindGroup(0, this.bindGroups.computeMorton);
            computePass.dispatchWorkgroups(dispatchCount);

            // sort morton codes and indices step
            this.pipelines.sortMorton.dispatch(computePass);

            // build LBVH step
            computePass.setPipeline(this.pipelines.buildLBVH);
            computePass.setBindGroup(0, this.bindGroups.buildLBVH);
            computePass.dispatchWorkgroups(dispatchCount);

            // fill LBVH step
            computePass.setPipeline(this.pipelines.fillLBVH);
            computePass.setBindGroup(0, this.bindGroups.fillLBVH);
            computePass.dispatchWorkgroups(dispatchCount);

            // barnes-hut velocity step
            computePass.setPipeline(this.pipelines.barnesHutVelStep);
            computePass.setBindGroup(0, this.bindGroups.barnesHutVelStep);
            computePass.dispatchWorkgroups(dispatchCount);

            // barnes-hut position step
            computePass.setPipeline(this.pipelines.barnesHutPosStep);
            computePass.setBindGroup(0, this.bindGroups.barnesHutPosStep);
            computePass.dispatchWorkgroups(dispatchCount);
        }

        computePass.end();
        return commandEncoder.finish();
    }

    public getBuffers(): SimBuffers {
        return this.buffers;
    }
    public getNumBodies(): number {
        return this.numBodies;
    }
}