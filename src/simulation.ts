import type { GPUCommandSource } from "./main";
import { createSimBindGroups, type SimBindGroups } from "./bindGroups";
import { createSimBuffers, type SimBuffers } from "./buffers";
import { getDefaultSimConfig, type SimConfig } from "./config";
import { createSimPipelines, type SimPipelines } from "./pipelines";


export class Simulation implements GPUCommandSource {
    // gpu device
    private readonly device: GPUDevice;

    // immutable config
    private readonly config: SimConfig;

    // dynamic state
    private camCenter: [number, number];
    private camHalfSize: [number, number];
    private viewPort: [number, number];

    // user-controlled body state
    private userBodyPos: [number, number];
    private userBodyMass: number;

    // GPU buffers, pipelines, and bind groups
    private buffers: SimBuffers;
    private pipelines: SimPipelines;
    private bindGroups: SimBindGroups;


    public constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
        // set up device
        this.device = device;

        // set up default config
        this.config = getDefaultSimConfig(canvas);

        // set up initial camera state
        this.camCenter = [0.0, 0.0];
        this.camHalfSize = [10.0, 10.0];
        this.viewPort = this.config.viewPort;
        
        // set up initial user body state
        this.userBodyPos = [0.0, 0.0];
        this.userBodyMass = 0.0;

        // set up GPU buffers, pipelines, and bind groups
        this.buffers = createSimBuffers(this.device, this.config, this.camCenter, this.camHalfSize, this.viewPort);
        this.pipelines = createSimPipelines(this.device, this.config, this.buffers);
        this.bindGroups = createSimBindGroups(this.device, this.buffers, this.pipelines);
    }

    public getCommands(): GPUCommandBuffer {
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();

        const workgroupSize = 64;
        const dispatchCount = Math.ceil(this.config.numBodies / workgroupSize);

        for (let step = 0; step < this.config.substeps; step++) {
            // compute morton codes step
            computePass.setPipeline(this.pipelines.computeMorton);
            computePass.setBindGroup(0, this.bindGroups.computeMortonStep);
            computePass.dispatchWorkgroups(dispatchCount);

            // sort morton codes and indices step
            this.pipelines.sortMortonCodes.dispatch(computePass);

            // build LBVH step
            computePass.setPipeline(this.pipelines.buildLBVH);
            computePass.setBindGroup(0, this.bindGroups.buildLBVHStep);
            computePass.dispatchWorkgroups(dispatchCount);

            // fill LBVH step
            computePass.setPipeline(this.pipelines.fillLBVH);
            computePass.setBindGroup(0, this.bindGroups.fillLBVHStep);
            computePass.dispatchWorkgroups(dispatchCount);

            // barnes-hut velocity step
            computePass.setPipeline(this.pipelines.barnesHutVelStep);
            computePass.setBindGroup(0, this.bindGroups.barnesHutVelStep);
            computePass.dispatchWorkgroups(dispatchCount);

            // barnes-hut position step
            computePass.setPipeline(this.pipelines.barnesHutPosStep);
            computePass.setBindGroup(0, this.bindGroups.barnesHutPosStep);
            computePass.dispatchWorkgroups(dispatchCount);


            // // half velocity step
            // computePass.setPipeline(this.pipelines.halfVelStep);
            // computePass.setBindGroup(0, this.bindGroups.halfVelStep);
            // computePass.dispatchWorkgroups(dispatchCount);

            // // position step
            // computePass.setPipeline(this.pipelines.posStep);
            // computePass.setBindGroup(0, this.bindGroups.posStep);
            // computePass.dispatchWorkgroups(dispatchCount);

            // // half velocity step
            // computePass.setPipeline(this.pipelines.halfVelStep);
            // computePass.setBindGroup(0, this.bindGroups.halfVelStep);
            // computePass.dispatchWorkgroups(dispatchCount);
        }

        computePass.end();
        return commandEncoder.finish();
    }

    public getBuffers(): SimBuffers {
        return this.buffers;
    }
    public getConfig(): SimConfig {
        return this.config;
    }
    public getCamCenter(): [number, number] {
        return this.camCenter;
    }
    public getCamHalfSize(): [number, number] {
        return this.camHalfSize;
    }
    public getViewPort(): [number, number] {
        return this.config.viewPort;
    }
    public getUserBodyPos(): [number, number] {
        return this.userBodyPos;
    }
    public getUserBodyMass(): number {
        return this.userBodyMass;
    }
    public setUserBodyMass(mass: number): void {
        this.userBodyMass = mass;
    }
    
}