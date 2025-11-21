import computeShaderCode from "./shaders/compute.wgsl?raw";
import renderShaderCode from "./shaders/render.wgsl?raw";


// Initialize WebGPU
const adapter = (await navigator.gpu.requestAdapter())!;
const device = await adapter.requestDevice();

// Setup canvas context
const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
const context = canvas.getContext("webgpu") as GPUCanvasContext;
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format: canvasFormat, alphaMode: "premultiplied" });

// Canvas and data dimensions
const dpr = window.devicePixelRatio || 1;
const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));



// ----- Initialize GPU buffers -----

// 1) Metadata buffers
const uintMetadata = new Uint32Array([width, height, 10]); // width, height, num_iters
const floatMetadata = new Float32Array([0.016, 1.5]); // delta_time, over_relaxation

const uintMetadataBuffer = device.createBuffer({
  size: uintMetadata.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uintMetadataBuffer, 0, uintMetadata.buffer, uintMetadata.byteOffset, uintMetadata.byteLength);

const floatMetadataBuffer = device.createBuffer({
  size: floatMetadata.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(floatMetadataBuffer, 0, floatMetadata.buffer, floatMetadata.byteOffset, floatMetadata.byteLength);


// 2) Velocity buffers (u and v components)
const uVelocity = new Float32Array((width + 1) * height).fill(10.0);
const vVelocity = new Float32Array(width * (height + 1)).fill(0.0);

// TEMP: initialize a rightward flow in the u velocity
// for (let y = 0; y < height; ++y) {
//   const left_x = 10;
//   const right_x = 30;
//   for (let x = left_x; x < right_x; ++x) {
//     uVelocity[y * (width + 1) + x] = 30.0;
//   }
// }

const newUVelocity = new Float32Array((width + 1) * height).fill(0.0);
const newVVelocity = new Float32Array(width * (height + 1)).fill(0.0);

const uBuffer = device.createBuffer({
  size: uVelocity.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uBuffer, 0, uVelocity.buffer, uVelocity.byteOffset, uVelocity.byteLength);

const vBuffer = device.createBuffer({
  size: vVelocity.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vBuffer, 0, vVelocity.buffer, vVelocity.byteOffset, vVelocity.byteLength);

const newUBuffer = device.createBuffer({
  size: newUVelocity.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(newUBuffer, 0, newUVelocity.buffer, newUVelocity.byteOffset, newUVelocity.byteLength);

const newVBuffer = device.createBuffer({
  size: newVVelocity.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(newVBuffer, 0, newVVelocity.buffer, newVVelocity.byteOffset, newVVelocity.byteLength);


// 3) Cell center buffers
function createCellCenterBuffer(initialData: Float32Array): GPUBuffer {
  const buffer = device.createBuffer({
    size: initialData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buffer, 0, initialData.buffer, initialData.byteOffset, initialData.byteLength);
  return buffer;
}

const dye = new Float32Array(width * height).fill(0.0);
const newDye = new Float32Array(width * height).fill(0.0);

// TEMP: initialize a checkerboard dye pattern
const checkerSize = 50;
for (let y = 0; y < height; ++y) {
  for (let x = 0; x < width; ++x) {
    const checkerX = Math.floor(x / checkerSize);
    const checkerY = Math.floor(y / checkerSize);
    if ((checkerX + checkerY) % 2 === 0) {
      dye[y * width + x] = 1.0;
    }
  }
}

const dyeBuffer = createCellCenterBuffer(dye);
const newDyeBuffer = createCellCenterBuffer(newDye);

const pressure = new Float32Array(width * height).fill(0.0);
const newPressure = new Float32Array(width * height).fill(0.0);
const pressureBuffer = createCellCenterBuffer(pressure);
const newPressureBuffer = createCellCenterBuffer(newPressure);

const divergence = new Float32Array(width * height).fill(0.0);
const divergenceBuffer = createCellCenterBuffer(divergence);

const obstacles = new Float32Array(width * height).fill(0);
// set borders as obstacles
for (let y = 0; y < height; ++y) {
  obstacles[y * width + 0] = 1.0;
  obstacles[y * width + (width - 1)] = 1.0;
}
for (let x = 0; x < width; ++x) {
  obstacles[0 * width + x] = 1.0;
  obstacles[(height - 1) * width + x] = 1.0;
}
// add a central square obstacle
const obstacleSize = 100;
const startX = Math.floor((width - obstacleSize) / 2);
const startY = Math.floor((height - obstacleSize) / 2);
for (let y = startY; y < startY + obstacleSize; ++y) {
  for (let x = startX; x < startX + obstacleSize; ++x) {
    obstacles[y * width + x] = 1.0;
  }
}
const obstaclesBuffer = createCellCenterBuffer(obstacles);


// ----- Create compute/graphics pipelines -----
const computeShaderModule = device.createShaderModule({ code: computeShaderCode });
const renderShaderModule = device.createShaderModule({ code: renderShaderCode });

// Advection pipeline
const advectionPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "advect_main",
  },
});


const advectionBindGroup = device.createBindGroup({
  layout: advectionPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: uBuffer } },
    { binding: 3, resource: { buffer: vBuffer } },
    { binding: 4, resource: { buffer: newUBuffer } },
    { binding: 5, resource: { buffer: newVBuffer } },
    // { binding: 6, resource: { buffer: pressureBuffer } },
    // { binding: 7, resource: { buffer: newPressureBuffer } },
    // { binding: 8, resource: { buffer: divergenceBuffer } },
    // { binding: 9, resource: { buffer: obstaclesBuffer } },
    { binding: 10, resource: { buffer: dyeBuffer } },
    { binding: 11, resource: { buffer: newDyeBuffer } },
  ],
});


// Divergence pipeline
const divergencePipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "divergence_main",
  },
});

const divergenceBindGroup = device.createBindGroup({
  layout: divergencePipeline.getBindGroupLayout(0),
  entries: [
    // { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: newUBuffer } },
    { binding: 3, resource: { buffer: newVBuffer } },
    // { binding: 4, resource: { buffer: newUBuffer } },
    // { binding: 5, resource: { buffer: newVBuffer } },
    // { binding: 6, resource: { buffer: newPressureBuffer } },
    // { binding: 7, resource: { buffer: pressureBuffer } },
    { binding: 8, resource: { buffer: divergenceBuffer } },
    { binding: 9, resource: { buffer: obstaclesBuffer } },
    // { binding: 10, resource: { buffer: newDyeBuffer } },
    // { binding: 11, resource: { buffer: dyeBuffer } },
  ],
});


// Pressure solve pipeline
const pressureSolvePipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "pressure_solve_main",
  },
});

const pressureSolveBindGroupA = device.createBindGroup({
  layout: pressureSolvePipeline.getBindGroupLayout(0),
  entries: [
    // { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    // { binding: 2, resource: { buffer: uBuffer } },
    // { binding: 3, resource: { buffer: vBuffer } },
    // { binding: 4, resource: { buffer: newUBuffer } },
    // { binding: 5, resource: { buffer: newVBuffer } },
    { binding: 6, resource: { buffer: pressureBuffer } },
    { binding: 7, resource: { buffer: newPressureBuffer } },
    { binding: 8, resource: { buffer: divergenceBuffer } },
    { binding: 9, resource: { buffer: obstaclesBuffer } },
    // { binding: 10, resource: { buffer: dyeBuffer } },
    // { binding: 11, resource: { buffer: newDyeBuffer } },
  ]
});

const pressureSolveBindGroupB = device.createBindGroup({
  layout: pressureSolvePipeline.getBindGroupLayout(0),
  entries: [
    // { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    // { binding: 2, resource: { buffer: uBuffer } },
    // { binding: 3, resource: { buffer: vBuffer } },
    // { binding: 4, resource: { buffer: newUBuffer } },
    // { binding: 5, resource: { buffer: newVBuffer } },
    { binding: 6, resource: { buffer: newPressureBuffer } },
    { binding: 7, resource: { buffer: pressureBuffer } },
    { binding: 8, resource: { buffer: divergenceBuffer } },
    { binding: 9, resource: { buffer: obstaclesBuffer } },
    // { binding: 10, resource: { buffer: dyeBuffer } },
    // { binding: 11, resource: { buffer: newDyeBuffer } },
  ]
});


// Projection pipeline
const projectionPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "project_main",
  },
});

const projectionBindGroup = device.createBindGroup({
  layout: projectionPipeline.getBindGroupLayout(0),
  entries: [
    // { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: newUBuffer } },
    { binding: 3, resource: { buffer: newVBuffer } },
    { binding: 4, resource: { buffer: uBuffer } },
    { binding: 5, resource: { buffer: vBuffer } },
    { binding: 6, resource: { buffer: pressureBuffer } },
    // { binding: 7, resource: { buffer: newPressureBuffer } },
    // { binding: 8, resource: { buffer: divergenceBuffer } },
    { binding: 9, resource: { buffer: obstaclesBuffer } },
    // { binding: 10, resource: { buffer: dyeBuffer } },
    // { binding: 11, resource: { buffer: newDyeBuffer } },
  ]
});


// Cleanup pipeline
const cleanupPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "cleanup_main",
  },
});

const cleanupBindGroup = device.createBindGroup({
  layout: cleanupPipeline.getBindGroupLayout(0),
  entries: [
    // { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    // { binding: 2, resource: { buffer: newUBuffer } },
    // { binding: 3, resource: { buffer: newVBuffer } },
    // { binding: 4, resource: { buffer: uBuffer } },
    // { binding: 5, resource: { buffer: vBuffer } },
    // { binding: 6, resource: { buffer: pressureBuffer } },
    // { binding: 7, resource: { buffer: newPressureBuffer } },
    // { binding: 8, resource: { buffer: divergenceBuffer } },
    // { binding: 9, resource: { buffer: obstaclesBuffer } },
    { binding: 10, resource: { buffer: newDyeBuffer } },
    { binding: 11, resource: { buffer: dyeBuffer } },
  ]
});


// Render pipeline
const renderPipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: {
    module: renderShaderModule,
    entryPoint: "vertex_main",
  },
  fragment: {
    module: renderShaderModule,
    entryPoint: "fragment_main",
    targets: [{ format: canvasFormat }],
  },
  primitive: {
    topology: "triangle-list",
  },
});

const renderBindGroup = device.createBindGroup({
  layout: renderPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: dyeBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
  ],
});



// ----- Main simulation loop -----
const numJacobiIterations = 10;

const workgroupX = 16;
const workgroupY = 16;

const dispatchX = Math.ceil((width + 1) / workgroupX);
const dispatchY = Math.ceil((height + 1) / workgroupY);

function frame() {
  const commandEncoder = device.createCommandEncoder();

  // Compute pass
  {
    const computePass = commandEncoder.beginComputePass();

    // 1) Advection
    computePass.setPipeline(advectionPipeline);
    computePass.setBindGroup(0, advectionBindGroup); // u, v, dye -> new u, new v, new dye
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    // 2) Divergence
    computePass.setPipeline(divergencePipeline);
    computePass.setBindGroup(0, divergenceBindGroup); // new u, new v -> divergence
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    // 3) Pressure solve (Jacobi iterations)
    computePass.setPipeline(pressureSolvePipeline);
    for (let i = 0; i < numJacobiIterations / 2; ++i) {
      if (i % 2 === 0) {
        computePass.setBindGroup(0, pressureSolveBindGroupA); // pressure, divergence -> new pressure
        computePass.dispatchWorkgroups(dispatchX, dispatchY);
        computePass.setBindGroup(0, pressureSolveBindGroupB); // new_pressure, divergence -> pressure
        computePass.dispatchWorkgroups(dispatchX, dispatchY);
      }
    }

    // 4) Projection
    computePass.setPipeline(projectionPipeline);
    computePass.setBindGroup(0, projectionBindGroup); // new_u, new_v, pressure -> corrected u, v
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    // 5) Cleanup / copy new buffers to current buffers
    computePass.setPipeline(cleanupPipeline);
    computePass.setBindGroup(0, cleanupBindGroup);
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    computePass.end();
  }

  // Render pass
  {
    const textureView = context.getCurrentTexture().createView();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(3);
    renderPass.end();
  }

  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);