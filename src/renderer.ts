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

export class Renderer {
    // gpu device and canvas context
    private readonly device: GPUDevice;
    private readonly canvas: HTMLCanvasElement;
    private readonly context: GPUCanvasContext;
    private readonly canvasFormat: GPUTextureFormat;

    // local state
    private viewPort: [number, number];
    private camCenter: [number, number];
    private camHalfSize: [number, number];
    private numBodies: number;

    // GPU buffers, pipelines, and bind groups
    private buffers: RenderBuffers;
    private pipelines: RenderPipelines;
    private bindGroups: RenderBindGroups;


    // INITIALIZATION

    public constructor(device: GPUDevice, canvas: HTMLCanvasElement, context: GPUCanvasContext, canvasFormat: GPUTextureFormat, initialPosBuffer: GPUBuffer) {
        this.device = device;
        this.canvas = canvas;
        this.context = context;
        this.canvasFormat = canvasFormat;

        this.viewPort = [context.canvas.width, context.canvas.height];
        this.camCenter = [0.0, 0.0];
        this.camHalfSize = [10.0, 10.0];
        this.numBodies = 0;

        // GPU buffers, pipelines, and bind groups
        this.buffers = this.createRenderBuffers();
        this.pipelines = this.createRenderPipelines();
        this.bindGroups = this.createRenderBindGroups(initialPosBuffer);

        // initial resize
        this.resizeCanvasToDisplaySize();
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

    private createRenderBindGroups(posBuffer: GPUBuffer): RenderBindGroups {
        const density = this.device.createBindGroup({
            layout: this.pipelines.density.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.metadataBuffer } },
                { binding: 1, resource: { buffer: posBuffer } },
            ],
        });

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
        uintView[6] = this.numBodies;

        this.device.queue.writeBuffer(this.buffers.metadataBuffer, 0, metadataArray);
    }

    private updateDensityTexture() {
        // recreate density texture with new size
        this.buffers.densityTexture.destroy();
        this.buffers.densityTexture = this.device.createTexture({
            size: [this.viewPort[0], this.viewPort[1]],
            format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.buffers.densityTextureView = this.buffers.densityTexture.createView();

        // recreate tone map bind group with new texture view
        this.bindGroups.toneMap = this.device.createBindGroup({
            layout: this.pipelines.toneMap.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.buffers.densityTextureView },
                { binding: 1, resource: this.buffers.densityTextureSampler },
            ],
        });
    }

    public rebindPosBuffer(posBuffer: GPUBuffer) {
        // recreate density bind group with new position buffer
        this.bindGroups.density = this.device.createBindGroup({
            layout: this.pipelines.density.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.metadataBuffer } },
                { binding: 1, resource: { buffer: posBuffer } },
            ],
        });
    }

    public setNumBodies(numBodies: number) {
        this.numBodies = numBodies;
        this.updateMetadataBuffer();
    }

    public resizeCanvasToDisplaySize() {
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

        // update viewport and cam aspect ratio
        this.viewPort = [displayWidth, displayHeight];
        const aspect = displayWidth / displayHeight;
        this.camHalfSize[0] = this.camHalfSize[1] * aspect;
        this.updateMetadataBuffer();

        // resize density texture
        this.updateDensityTexture();
    }

    public deltaPxToDeltaWorld(deltaPx: number, deltaPy: number): [number, number] {
        const deltaNdcX = (2.0 * deltaPx) / this.viewPort[0];
        const deltaNdcY = (2.0 * deltaPy) / this.viewPort[1];

        const deltaWorldX = deltaNdcX * this.camHalfSize[0];
        const deltaWorldY = deltaNdcY * this.camHalfSize[1];

        return [deltaWorldX, deltaWorldY];
    }

    public canvasPxToWorld(px: number, py: number): [number, number] {
        const ndcX = (2.0 * px) / this.viewPort[0] - 1.0;
        const ndcY = (2.0 * py) / this.viewPort[1] - 1.0;

        const worldX = this.camCenter[0] + ndcX * this.camHalfSize[0];
        const worldY = this.camCenter[1] + ndcY * this.camHalfSize[1];

        return [worldX, worldY];
    }

    public panCamera(deltaPx: number, deltaPy: number) {
        const [deltaWorldX, deltaWorldY] = this.deltaPxToDeltaWorld(deltaPx, deltaPy);
        this.camCenter[0] -= deltaWorldX;
        this.camCenter[1] -= deltaWorldY;
        this.updateMetadataBuffer();
    }

    public zoomCamera(zoomFactor: number, zoomCenterX: number, zoomCenterY: number) {
        const [worldZoomCenterX, worldZoomCenterY] = this.canvasPxToWorld(zoomCenterX, zoomCenterY);

        this.camHalfSize[0] *= zoomFactor;
        this.camHalfSize[1] *= zoomFactor;

        // adjust cam center to zoom towards zoom center
        this.camCenter[0] = this.camCenter[0] + (worldZoomCenterX - this.camCenter[0]) * (1.0 - zoomFactor);
        this.camCenter[1] = this.camCenter[1] + (worldZoomCenterY - this.camCenter[1]) * (1.0 - zoomFactor);

        this.updateMetadataBuffer();
    }


    public getCommands(numBodies: number): GPUCommandBuffer {
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
        densityPass.draw(6, numBodies, 0, 0);
        densityPass.end();


        // second pass: render to screen from density texture
        const toneMapPass = commandEncoder.beginRenderPass({
            colorAttachments: [
            {
                view: canvasTextureView,
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: "clear",
                storeOp: "store",
            },
            ],
        });
        toneMapPass.setPipeline(this.pipelines.toneMap);
        toneMapPass.setBindGroup(0, this.bindGroups.toneMap);
        toneMapPass.draw(3, 1, 0, 0);
        toneMapPass.end();

        return commandEncoder.finish();
    }
}