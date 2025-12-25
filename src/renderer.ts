import { createRenderBindGroups, type RenderBindGroups } from "./bindGroups";
import { createRenderBuffers, type RenderBuffers } from "./buffers";
import type { GPUCommandSource } from "./main";
import { createRenderPipelines, type RenderPipelines } from "./pipelines";
import type { Simulation } from "./simulation";


export class Renderer implements GPUCommandSource {
    // gpu device and canvas context
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;

    // sim instance
    private readonly sim: Simulation;

    // GPU buffers, pipelines, and bind groups
    private buffers: RenderBuffers;
    private pipelines: RenderPipelines;
    private bindGroups: RenderBindGroups;

    public constructor(device: GPUDevice, context: GPUCanvasContext, sim: Simulation, canvasFormat: GPUTextureFormat) {
        this.device = device;
        this.context = context;

        this.sim = sim;

        // GPU buffers, pipelines, and bind groups
        this.buffers = createRenderBuffers(device, sim.getBuffers(), sim.getViewPort());
        this.pipelines = createRenderPipelines(device, canvasFormat);
        this.bindGroups = createRenderBindGroups(device, this.buffers, this.pipelines);
    }

    public getCommands(): GPUCommandBuffer {
        const commandEncoder = this.device.createCommandEncoder();
        const canvasTextureView = this.context.getCurrentTexture().createView();

        // first pass: fill density texture
        const densityPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.buffers.densityTextureView,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: "clear",
                storeOp: "store",
            }],
        });
        densityPass.setPipeline(this.pipelines.density);
        densityPass.setBindGroup(0, this.bindGroups.density);
        densityPass.draw(6, this.sim.getNumBodies(), 0, 0);
        densityPass.end();


        // second pass: render to screen from density texture
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [
            {
                view: canvasTextureView,
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: "clear",
                storeOp: "store",
            },
            ],
        });
        renderPass.setPipeline(this.pipelines.toneMap);
        renderPass.setBindGroup(0, this.bindGroups.toneMap);
        renderPass.draw(3, 1, 0, 0);
        renderPass.end();

        return commandEncoder.finish();
    }

    public updateBuffers(sim: Simulation): void {
        this.buffers = createRenderBuffers(this.device, sim.getBuffers(), sim.getViewPort());
        this.pipelines = createRenderPipelines(this.device, this.context.getCurrentTexture().format);
        this.bindGroups = createRenderBindGroups(this.device, this.buffers, this.pipelines);
    }
}