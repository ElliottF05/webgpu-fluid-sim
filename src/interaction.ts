import { buildMetadataArrays } from "./buffers";
import type { Simulation } from "./simulation";


export class InteractionController {
    private readonly device: GPUDevice;
    private readonly format: GPUTextureFormat;
    private readonly canvas: HTMLCanvasElement;
    private readonly context: GPUCanvasContext;

    private readonly sim: Simulation;

    private interactionMode: "camera" | "body" = "camera";
    private userBodySliderValue: number = 0.0;
    private isMouseDown: boolean = false;
    private lastMousePos: [number, number] = [0, 0];
    
    public constructor(device: GPUDevice, format: GPUTextureFormat, canvas: HTMLCanvasElement, context: GPUCanvasContext, sim: Simulation) {
        this.device = device;
        this.format = format;
        this.canvas = canvas;
        this.context = context;
        this.sim = sim;

        // set up event listeners
        this.addScrollListener();
        this.addDragListener();
        this.addResizeListener();
        this.addUserMassListeners();
        this.addNumBodiesListeners();
    }

    private getMouseWorldPos(clientX: number, clientY: number): [number, number] {
        const rect = this.canvas.getBoundingClientRect();

        const [camCenterX, camCenterY] = this.sim.getCamCenter();
        const [camHalfSizeX, camHalfSizeY] = this.sim.getCamHalfSize();

        // pixel coords relative to canvas
        const px = clientX - rect.left;
        const py = clientY - rect.top;

        // normalized to [-1,1]
        const u = (2 * px / rect.width) - 1;
        const v = (2 * py / rect.height) - 1;

        // to world pos
        const worldX = u * camHalfSizeX + camCenterX;
        const worldY = v * camHalfSizeY + camCenterY;

        return [worldX, worldY];
    }

    private clamp(value: number, min: number, max: number): number {
        return Math.min(max, Math.max(min, value));
    }

    private addScrollListener() {
        // Scrolling for zooming
        this.canvas.addEventListener("wheel", (e) => {
        e.preventDefault();

        // position before zoom
        const [x1, y1] = this.getMouseWorldPos(e.clientX, e.clientY);
        const zoomSpeed = 0.0015;
        const zoomFactor = Math.exp(e.deltaY * zoomSpeed);

        // clamp
        const minHalfSize = 0.01;
        const maxHalfSize = 1e4;

        const camCenter = this.sim.getCamCenter();
        const camHalfSize = this.sim.getCamHalfSize();
        const viewPort = this.sim.getViewPort();

        camHalfSize[0] = this.clamp(camHalfSize[0] * zoomFactor, minHalfSize, maxHalfSize);

        // keep aspect ratio
        const aspect = viewPort[0] / viewPort[1];
        camHalfSize[1] = camHalfSize[0] / aspect;

        // get position after zoom
        const [x2, y2] = this.getMouseWorldPos(e.clientX, e.clientY);

        // shift camera
        camCenter[0] += x1 - x2;
        camCenter[1] += y1 - y2;
        }, {passive: false})
    }

    private addDragListener() {
        const setUserBodyPos = (clientX: number, clientY: number) => {
            const [worldX, worldY] = this.getMouseWorldPos(clientX, clientY);
            const userBodyPos = this.sim.getUserBodyPos();
            userBodyPos[0] = worldX;
            userBodyPos[1] = worldY;
            this.sim.setUserBodyMass(this.userBodySliderValue);
        }

        this.canvas.addEventListener("pointerdown", (e) => {
            if (e.button !== 0) {
                return; // must be left moust button
            }
            this.isMouseDown = true;
            this.lastMousePos = [e.clientX, e.clientY];
            this.canvas.setPointerCapture(e.pointerId);

            if (this.interactionMode === "body") {
                setUserBodyPos(e.clientX, e.clientY);
            }
        });

        this.canvas.addEventListener("pointermove", (e) => {
            if (!this.isMouseDown) {
                return;
            }
            
            if (this.interactionMode === "body") {
                setUserBodyPos(e.clientX, e.clientY);
            } else {
                const rect = this.canvas.getBoundingClientRect();
                const dxPixels = e.clientX - this.lastMousePos[0];
                const dyPixels = e.clientY - this.lastMousePos[1];
                this.lastMousePos = [e.clientX, e.clientY];

                const camCenter = this.sim.getCamCenter();
                const camHalfSize = this.sim.getCamHalfSize();

                const worldPerPixelX = (2 * camHalfSize[0]) / rect.width;
                const worldPerPixelY = (2 * camHalfSize[1]) / rect.height;

                camCenter[0] -= dxPixels * worldPerPixelX;
                camCenter[1] -= dyPixels * worldPerPixelY
            }
        });

        const endPan = (e: PointerEvent) => {
            if (!this.isMouseDown) {
                return;
            }
            this.isMouseDown = false;
            try {
                this.canvas.releasePointerCapture(e.pointerId);
            } catch {
                // do nothing
            }
            if (this.interactionMode === "body") {
                this.sim.setUserBodyMass(0.0);
            }
        }

        this.canvas.addEventListener("pointerup", endPan);
        this.canvas.addEventListener("pointercancel", endPan);
    }

    private addUserMassListeners() {
        const modeRadios = document.getElementsByName("mode") as NodeListOf<HTMLInputElement>;
        modeRadios.forEach(radio => {
            radio.addEventListener("change", (_e) => {
                if (radio.checked) {
                    this.interactionMode = radio.value as "camera" | "body";
                }
            });
        });

        const massSlider = document.getElementById("massSlider") as HTMLInputElement;
        const massValue = document.getElementById("massValue") as HTMLSpanElement;

        massSlider.addEventListener("input", (_e) => {
            const sliderVal = parseFloat(massSlider.value);
            const a = 10; // larger => steeper ends, flatter center
            const value = Math.sinh(a * sliderVal) / Math.sinh(a) * 100_000;
            massValue.textContent = value.toFixed(1);
            this.userBodySliderValue = value;
        });

        window.addEventListener("keydown", (e) => {
            if (e.key === "v") {
                if (this.interactionMode === "body") {
                    this.interactionMode = "camera";
                    this.sim.setUserBodyMass(0.0);
                } else {
                    this.interactionMode = "body";
                    this.sim.setUserBodyMass(this.userBodySliderValue);
                }
            }
        });
    }

    private addNumBodiesListeners() {
        const numBodiesSelect = document.getElementById("numBodiesSelect") as HTMLInputElement;
        numBodiesSelect.addEventListener("change", () => {
            const numBodies = parseInt(numBodiesSelect.value, 10);
            if (Number.isNaN(numBodies) || numBodies <= 0) {
                return;
            }
            this.sim.setNumBodies(numBodies);
        });
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
                format: this.format,
                alphaMode: "premultiplied",
            });
        }

        return { displayWidth, displayHeight, dpr };
    }

    private onResize(): void {
        const { displayWidth, displayHeight } = this.resizeCanvasToDisplaySize();
        const viewPort = this.sim.getViewPort();
        viewPort[0] = displayWidth;
        viewPort[1] = displayHeight;
        
        // adjust camera half size to maintain aspect ratio
        const camHalfSize = this.sim.getCamHalfSize();
        const aspect = displayWidth / displayHeight;
        camHalfSize[0] = camHalfSize[1] * aspect;
    }

    private addResizeListener() {
        window.addEventListener("resize", () => {
            this.onResize();
        });
        // call once to set initial size
        this.onResize();
    }

    public sendUpdateToGPU(): void {
        // sync metadata buffers
        const { uintMetadataArray, floatMetadataArray } = buildMetadataArrays(
            this.sim.getConfig(),
            this.sim.getNumBodies(),
            this.sim.getCamCenter(),
            this.sim.getCamHalfSize(),
            this.sim.getViewPort()
        );

        this.device.queue.writeBuffer(
            this.sim.getBuffers().uintMetadata,
            0,
            uintMetadataArray.buffer,
            uintMetadataArray.byteOffset,
            uintMetadataArray.byteLength
        );

        this.device.queue.writeBuffer(
            this.sim.getBuffers().floatMetadata,
            0,
            floatMetadataArray.buffer,
            floatMetadataArray.byteOffset,
            floatMetadataArray.byteLength
        );

        // sync user body mass and pos
        const userBodyMassArray = new Float32Array([this.sim.getUserBodyMass()]);
        this.device.queue.writeBuffer(
            this.sim.getBuffers().mass,
            0,
            userBodyMassArray.buffer,
            userBodyMassArray.byteOffset,
            userBodyMassArray.byteLength
        );

        const userBodyPos = this.sim.getUserBodyPos();
        const userBodyPosArray = new Float32Array([userBodyPos[0], userBodyPos[1]]);
        this.device.queue.writeBuffer(
            this.sim.getBuffers().pos,
            0,
            userBodyPosArray.buffer,
            userBodyPosArray.byteOffset,
            userBodyPosArray.byteLength
        );
    }
}