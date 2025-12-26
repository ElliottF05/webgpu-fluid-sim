// import { readBufferData } from "./buffers";
import { getDefaultConfig } from "./config";
import { initDeviceAndContext } from "./gpuSetup";
import { InteractionController } from "./interaction";
import { Renderer } from "./renderer";
import { Simulation } from "./simulation";

const TARGET_FPS = 60;
const FRAME_DURATION_MS = 1000.0 / TARGET_FPS;

export interface GPUCommandSource {
    getCommands(): GPUCommandBuffer;
}

async function main() {
    const config = getDefaultConfig();
    const { device, context, canvasFormat, canvas } = await initDeviceAndContext("webgpu-canvas");

    const sim = new Simulation(config, device);
    const renderer = new Renderer(device, canvas, context, canvasFormat, sim);
    // sim.setRenderer(renderer); might not need this
    // const interaction = new InteractionController(device, canvasFormat, canvas, context, sim);

    let last = performance.now();
    function frame() {
        requestAnimationFrame(frame);

        const now = performance.now();
        if (now - last < FRAME_DURATION_MS) {
            return;
        }
        last = now;

        // interaction.sendUpdateToGPU();
        device.queue.submit([
            sim.getCommands(),
            renderer.getCommands(),
        ]);

        device.queue.onSubmittedWorkDone().then(() => {
            console.log(`Frame time: ${(performance.now() - now).toFixed(2)} ms`);
        });
    }

    requestAnimationFrame(frame);
}

await main();