export async function initDeviceAndContext(canvasId: string) {
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    const adapter = (await navigator.gpu.requestAdapter())!;

    const device = await adapter.requestDevice();
    const context = canvas.getContext("webgpu") as GPUCanvasContext;
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    context.configure({
        device: device,
        format: canvasFormat,
        alphaMode: "premultiplied",
    });

    return { device, context, canvasFormat, canvas };
}