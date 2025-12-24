// import { readBufferData } from "./buffers";
import { initDeviceAndContext } from "./gpu";
import { InteractionController } from "./interaction";
import { Renderer } from "./renderer";
import { Simulation } from "./simulation";

export interface GPUCommandSource {
    getCommands(): GPUCommandBuffer;
}

async function main() {
    const { device, context, canvasFormat, canvas } = await initDeviceAndContext("webgpu-canvas");

    const sim = new Simulation(device, canvas);
    const renderer = new Renderer(device, context, sim.getConfig(),sim.getBuffers(), canvasFormat);
    const interaction = new InteractionController(device, canvasFormat, canvas, context, sim);

    function frame() {
        device.queue.submit([
            sim.getCommands(),
            renderer.getCommands(),
        ]);
        interaction.sendUpdateToGPU();
        requestAnimationFrame(frame);   
    }
    requestAnimationFrame(frame);
}

await main();