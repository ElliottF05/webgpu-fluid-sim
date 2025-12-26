import type { GPUCommandSource } from "./main";
import type { Simulation } from "./simulation";
import densityShaderCode from "./shaders/render/density.wgsl?raw";
import toneMapShaderCode from "./shaders/render/tone_map.wgsl?raw";


type RenderBuffers = {
    metadataBuffer: GPUBuffer;
    densityTexture: GPUTexture;
    densityTextureView: GPUTextureView;
    densityTextureSampler: GPUSampler;
}

type RenderPipelines = {
    density: GPURenderPipeline;
    toneMap: GPURenderPipeline;
}

type RenderBindGroups = {
    density: GPUBindGroup;
    toneMap: GPUBindGroup;
};

export class Renderer implements GPUCommandSource {
    // gpu device and canvas context
    private readonly device: GPUDevice;
    private readonly canvas: HTMLCanvasElement;
    private readonly context: GPUCanvasContext;
    private readonly canvasFormat: GPUTextureFormat;

    // sim instance
    private readonly sim: Simulation;

    // local state
    private viewPort: [number, number];
    private camCenter: [number, number];
    private camHalfSize: [number, number];

    // GPU buffers, pipelines, and bind groups
    private buffers: RenderBuffers;
    private pipelines: RenderPipelines;
    private bindGroups: RenderBindGroups;


    // INITIALIZATION

    public constructor(device: GPUDevice, canvas: HTMLCanvasElement, context: GPUCanvasContext, canvasFormat: GPUTextureFormat, sim: Simulation) {
        this.device = device;
        this.canvas = canvas;
        this.context = context;
        this.canvasFormat = canvasFormat;
        this.sim = sim;

        this.resizeCanvasToDisplaySize();

        this.viewPort = [context.canvas.width, context.canvas.height];
        this.camCenter = [0.0, 0.0];
        this.camHalfSize = [-10.0, 10.0];

        // GPU buffers, pipelines, and bind groups
        this.buffers = this.createRenderBuffers();
        this.pipelines = this.createRenderPipelines();
        this.bindGroups = this.createRenderBindGroups();

        // fill buffers with initial data
        this.updateMetadataBuffer();
    }

    private createRenderBuffers(): RenderBuffers {
        const metadataBuffer = this.device.createBuffer({
            size: 8 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const densityTexture = this.device.createTexture({
            size: [this.viewPort[0], this.viewPort[1]],
            format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        const densityTextureView = densityTexture.createView();
        const densityTextureSampler = this.device.createSampler({
            magFilter: "linear",
            minFilter: "linear",
        });

        return {
            metadataBuffer,
            densityTexture,
            densityTextureView,
            densityTextureSampler,
        }
    }

    private createRenderPipelines(): RenderPipelines {
        const densityShaderModule = this.device.createShaderModule({
            code: densityShaderCode,
        });
        const toneMapShaderModule = this.device.createShaderModule({
            code: toneMapShaderCode,
        });

        const densityPipeline = this.device.createRenderPipeline({
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
                    blend: {
                        color: { srcFactor: "one", dstFactor: "one", operation: "add" },
                        alpha: { srcFactor: "one", dstFactor: "one", operation: "add", },
                    }
                }],
            },
            primitive: {
                topology: "triangle-list",
            },
        });

        const toneMapPipeline = this.device.createRenderPipeline({
            layout: "auto",
            vertex: {
                module: toneMapShaderModule,
                entryPoint: "vertex_main",
            },
            fragment: {
                module: toneMapShaderModule,
                entryPoint: "fragment_main",
                targets: [{ 
                    format: this.canvasFormat,
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
            density: densityPipeline,
            toneMap: toneMapPipeline,
        };
    }

    private createRenderBindGroups(): RenderBindGroups {
        const density = this.device.createBindGroup({
            layout: this.pipelines.density.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.metadataBuffer } },
                { binding: 1, resource: { buffer: this.sim.getBuffers().pos } },
            ],
        });

        // adjust these as needed too
        const toneMap = this.device.createBindGroup({
            layout: this.pipelines.toneMap.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.buffers.densityTextureView },
                { binding: 1, resource: this.buffers.densityTextureSampler },
            ],
        });

        return {
            density,
            toneMap,
        };
    }

    private updateMetadataBuffer() {
        const metadataArray = new ArrayBuffer(8 * 4);
        const floatView = new Float32Array(metadataArray);
        const uintView = new Uint32Array(metadataArray);
        
        floatView[0] = this.camCenter[0];
        floatView[1] = this.camCenter[1];
        floatView[2] = this.camHalfSize[0];
        floatView[3] = this.camHalfSize[1];
        floatView[4] = this.viewPort[0];
        floatView[5] = this.viewPort[1];
        uintView[6] = this.sim.getNumBodies();

        this.device.queue.writeBuffer(this.buffers.metadataBuffer, 0, metadataArray);
    }

    private resizeCanvasToDisplaySize() {
        const dpr = window.devicePixelRatio || 1;

        const rect = this.canvas.getBoundingClientRect();
        const displayWidth = Math.max(1, Math.round(rect.width * dpr));
        const displayHeight = Math.max(1, Math.round(rect.height * dpr));

        if (this.canvas.width !== displayWidth || this.canvas.height !== displayHeight) {
            this.canvas.width = displayWidth;
            this.canvas.height = displayHeight;

            this.context.configure({
                device: this.device,
                format: this.canvasFormat,
                alphaMode: "premultiplied",
            });
        }

        return { displayWidth, displayHeight, dpr };
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
}