// import { readBufferData } from "./buffers";
import { getDefaultConfig } from "./config";
import { initDeviceAndContext } from "./gpuSetup";
import { InteractionController } from "./interaction";
import { Renderer } from "./renderer";
import { Simulation } from "./simulation";

const TARGET_FPS = 60;
const FRAME_DURATION_MS = 1000.0 / TARGET_FPS;

async function main() {
    const config = getDefaultConfig();
    const { device, context, canvasFormat, canvas } = await initDeviceAndContext("webgpu-canvas");

    const sim = new Simulation(config, device);
    const renderer = new Renderer(device, canvas, context, canvasFormat, sim.getBuffers().pos);
    // @ts-ignore 
    const interaction = new InteractionController(canvas, sim, renderer);

    // initial setup
    sim.setNumBodies(50000);
    sim.setScenario("default");
    renderer.rebindPosBuffer(sim.getBuffers().pos);
    renderer.setNumBodies(sim.getNumBodies());

    let lastFrame = performance.now();
    let lastLog = performance.now();
    let frameCount = 0;

    function frame() {
        requestAnimationFrame(frame);

        const now = performance.now();
        if (now - lastFrame < 0.9 * FRAME_DURATION_MS) {
            return;
        }
        lastFrame = now;

        // interaction.sendUpdateToGPU();
        device.queue.submit([
            sim.getCommands(),
            renderer.getCommands(sim.getNumBodies()),
        ]);
        sim.updateMetadataBuffer(); // temporal coupling, improve this later?
        
        frameCount++;
        if (now - lastLog >= 1000) {
            console.log(`FPS: ${frameCount}, Avg frame time: ${(1000 / frameCount).toFixed(2)} ms`);
            frameCount = 0;
            lastLog = now;
        }
    }

    requestAnimationFrame(frame);
}

await main();