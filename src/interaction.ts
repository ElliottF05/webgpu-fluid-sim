import type { Renderer } from "./renderer";
import type { SimScenario, Simulation } from "./simulation";


export class InteractionController {
    private readonly canvas: HTMLCanvasElement;
    private readonly sim: Simulation;
    private readonly renderer: Renderer;

    // interaction state
    private isMouseDown: boolean = false;
    private lastMouseCanvasPos: [number, number] = [0, 0];
    
    public constructor(canvas: HTMLCanvasElement, sim: Simulation, renderer: Renderer) {
        this.canvas = canvas;
        this.sim = sim;
        this.renderer = renderer;

        // set up event listeners
        this.addScrollListener();
        this.addDragListener();
        this.addResizeListener();
        this.addNumBodiesListeners();
        this.addScenarioListener();
    }

    // functions that coordinate updates affecting both simulation and renderer
    private updateNumBodies(numBodies: number) {
        this.sim.setNumBodies(numBodies);
        this.renderer.setNumBodies(this.sim.getNumBodies());
        this.renderer.rebindSimBuffers(this.sim.getBuffers().pos, this.sim.getBuffers().radiusMultiplier);
    }
    private updateScenario(scenario: SimScenario) {
        this.sim.setScenario(scenario);
        this.renderer.rebindSimBuffers(this.sim.getBuffers().pos, this.sim.getBuffers().radiusMultiplier);
    }

    private clientToCanvasCoords(clientX: number, clientY: number): [number, number] {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const canvasX = (clientX - rect.left) * dpr;
        const canvasY = (clientY - rect.top) * dpr;
        return [canvasX, canvasY];
    }

    private addScrollListener() {
        // Scrolling for zooming
        this.canvas.addEventListener("wheel", (e) => {
        e.preventDefault();

        const zoomSpeed = 0.0015;
        const zoomFactor = Math.exp(e.deltaY * zoomSpeed);

        const [canvasX, canvasY] = this.clientToCanvasCoords(e.clientX, e.clientY);
        this.renderer.zoomCamera(zoomFactor, canvasX, canvasY);
        }, {passive: false})
    }

    private addDragListener() {
        this.canvas.addEventListener("pointerdown", (e) => {
            if (e.button !== 0) {
                return; // must be left mouse button
            }
            this.isMouseDown = true;
            this.lastMouseCanvasPos = this.clientToCanvasCoords(e.clientX, e.clientY);
            this.canvas.setPointerCapture(e.pointerId);
        });

        this.canvas.addEventListener("pointermove", (e) => {
            if (!this.isMouseDown) {
                return;
            }
            const [canvasX, canvasY] = this.clientToCanvasCoords(e.clientX, e.clientY);
            const deltaX = canvasX - this.lastMouseCanvasPos[0];
            const deltaY = canvasY - this.lastMouseCanvasPos[1];
            this.lastMouseCanvasPos = [canvasX, canvasY];

            this.renderer.panCamera(deltaX, deltaY);
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
        }

        this.canvas.addEventListener("pointerup", endPan);
        this.canvas.addEventListener("pointercancel", endPan);
    }

    private addNumBodiesListeners() {
        const numBodiesSelect = document.getElementById("numBodiesSelect") as HTMLInputElement;
        numBodiesSelect.addEventListener("change", () => {
            const numBodies = parseInt(numBodiesSelect.value, 10);
            if (Number.isNaN(numBodies) || numBodies <= 0) {
                return;
            }
            this.updateNumBodies(numBodies);
        });
    }

    private addResizeListener() {
        window.addEventListener("resize", () => {
            this.renderer.resizeCanvasToDisplaySize();
        });
    }

    private addScenarioListener() {
        const scenarioSelect = document.getElementById("scenarioSelect") as HTMLSelectElement;
        scenarioSelect.addEventListener("change", () => {
            const scenario = scenarioSelect.value as SimScenario;
            this.updateScenario(scenario);
        });
    }
}