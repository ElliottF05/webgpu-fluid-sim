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

// Canvas dimensions
const width = canvas.width;
const height = canvas.height;

// Simulation parameters
const numBodies = 10;
const gravConstant = 1.0;
const substeps = 10;
const camCenter = [0.0, 0.0];
const camHalfSize = [10.0, 10.0];
const viewPort = [width, height]

// Derived values
const deltaTime = 1.0 / (60.0 * substeps)


// ----- Initialize GPU buffers -----

function createBuffer(initialData: Uint32Array | Float32Array): GPUBuffer {
  const buffer = device.createBuffer({
    size: initialData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buffer, 0, initialData.buffer, initialData.byteOffset, initialData.byteLength);
  return buffer;
}
function createMetadataBuffer(initialData: Uint32Array | Float32Array): GPUBuffer {
  const buffer = device.createBuffer({
    size: initialData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, initialData.buffer, initialData.byteOffset, initialData.byteLength);
  return buffer;
}

// Metadata buffers
const uintMetadata = new Uint32Array([numBodies]); // num_bodies
const floatMetadata = new Float32Array([gravConstant, deltaTime, ...camCenter, ...camHalfSize, ...viewPort]); // delta_time, cell_size, diffusion_rate, viscosity, vorticity

const uintMetadataBuffer = createMetadataBuffer(uintMetadata);
const floatMetadataBuffer = createMetadataBuffer(floatMetadata);


// Mass buffer
const mass = new Float32Array(numBodies).fill(1.0);
const massBuffer = createBuffer(mass);


// Pos buffers
const initPos = new Float32Array(2 * numBodies).fill(0.0);
for (let i = 0; i < numBodies; i++) {
  const x = (Math.random() - 0.5) * 10;
  const y = (Math.random() - 0.5) * 10;
  initPos[2*i] = x;
  initPos[2*i + 1] = y;
}
const posBuffer = createBuffer(initPos);


// Velocity buffers
const initVel = new Float32Array(2 * numBodies).fill(0.0);
for (let i = 0; i < numBodies; i++) {
  const x = (Math.random() - 1) * 1;
  const y = (Math.random() - 1) * 1;
  initVel[2*i] = x;
  initVel[2*i + 1] = y;
}
const velBuffer = createBuffer(initVel);

// ----- Create compute/graphics pipelines -----
const computeShaderModule = device.createShaderModule({ code: computeShaderCode });
const renderShaderModule = device.createShaderModule({ code: renderShaderCode });

// // metadata buffers
// @group(0) @binding(0) var<uniform> float_metadata: FloatMetadata;
// @group(0) @binding(1) var<uniform> uint_metadata: UintMetadata;

// // data buffers
// @group(0) @binding(2) var<storage, read_write> mass_buf: array<f32>;
// @group(0) @binding(3) var<storage, read_write> pos_buf: array<vec2<f32>>;
// @group(0) @binding(4) var<storage, read_write> vel_buf: array<vec2<f32>>;

// Compute pipelines
const halfVelocityStepPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "half_vel_step_main",
  },
})
const halfVelocityStepBindGroup = device.createBindGroup({
  layout: halfVelocityStepPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: massBuffer } },
    { binding: 3, resource: { buffer: posBuffer } },
    { binding: 4, resource: { buffer: velBuffer } },
  ]
})

const posStepPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "pos_step_main",
  },
})
const posStepBindGroup = device.createBindGroup({
  layout: posStepPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    // { binding: 2, resource: { buffer: massBuffer } },
    { binding: 3, resource: { buffer: posBuffer } },
    { binding: 4, resource: { buffer: velBuffer } },
  ]
})


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
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: posBuffer } },
  ],
});


// ----- Main simulation loop -----
const workgroupX = 16;
const dispatchX = Math.ceil(numBodies / workgroupX);

function frame() {
  const commandEncoder = device.createCommandEncoder();

  // Compute pass
  const computePass = commandEncoder.beginComputePass();
  for (let i = 0; i < substeps; i++) {
    computePass.setPipeline(halfVelocityStepPipeline);
    computePass.setBindGroup(0, halfVelocityStepBindGroup);
    computePass.dispatchWorkgroups(dispatchX);

    computePass.setPipeline(posStepPipeline);
    computePass.setBindGroup(0, posStepBindGroup);
    computePass.dispatchWorkgroups(dispatchX);

    computePass.setPipeline(halfVelocityStepPipeline);
    computePass.setBindGroup(0, halfVelocityStepBindGroup);
    computePass.dispatchWorkgroups(dispatchX);
  }
  
  computePass.end();

  // Render pass
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
  renderPass.draw(6, numBodies, 0, 0);
  renderPass.end();

  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);