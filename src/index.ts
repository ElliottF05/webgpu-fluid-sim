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
// const dpr = window.devicePixelRatio || 1;
// const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
// const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
const width = canvas.width;
const height = canvas.height;



// ----- Initialize GPU buffers -----

// 1) Metadata buffers
const uintMetadata = new Uint32Array([width, height]); // width, height
const floatMetadata = new Float32Array([0.016, 1.0, 0.0, 0.0]); // delta_time, cell_size, diffusion_rate, viscosity

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
function createBuffer(initialData: Uint32Array | Float32Array): GPUBuffer {
  const buffer = device.createBuffer({
    size: initialData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buffer, 0, initialData.buffer, initialData.byteOffset, initialData.byteLength);
  return buffer;
}

const uVelocity = new Float32Array(width * height).fill(0.0);
const vVelocity = new Float32Array(width * height).fill(0.0);

const newUVelocity = new Float32Array(width * height).fill(0.0);
const newVVelocity = new Float32Array(width * height).fill(0.0);

const uBuffer = createBuffer(uVelocity);
const vBuffer = createBuffer(vVelocity);
const newUBuffer = createBuffer(newUVelocity);
const newVBuffer = createBuffer(newVVelocity);


// 3) Density buffers
const density = new Float32Array(width * height).fill(0.0);
const newDensity = new Float32Array(width * height).fill(0.0);

const densityBuffer = createBuffer(density);
const newDensityBuffer = createBuffer(newDensity);


// 4) Divergence and pressure buffers
const divergence = new Float32Array(width * height).fill(0.0);
const pressure = new Float32Array(width * height).fill(0.0);

const divergenceBuffer = createBuffer(divergence);
const pressureBuffer = createBuffer(pressure);


// 5) Sources buffers
const densitySources = new Float32Array(width * height).fill(0.0);
const uSources = new Float32Array(width * height).fill(0.0);
const vSources = new Float32Array(width * height).fill(0.0);

const densitySourcesBuffer = createBuffer(densitySources);
const uSourcesBuffer = createBuffer(uSources);
const vSourcesBuffer = createBuffer(vSources);


// ----- Create compute/graphics pipelines -----
const computeShaderModule = device.createShaderModule({ code: computeShaderCode });
const renderShaderModule = device.createShaderModule({ code: renderShaderCode });

// // metadata buffers
// @group(0) @binding(0) var<uniform> float_metadata: FloatMetadata;
// @group(0) @binding(1) var<uniform> uint_metadata: UintMetadata;

// // velocity fields
// @group(0) @binding(2) var<storage, read_write> u: array<f32>;
// @group(0) @binding(3) var<storage, read_write> v: array<f32>;
// @group(0) @binding(4) var<storage, read_write> u_new: array<f32>;
// @group(0) @binding(5) var<storage, read_write> v_new: array<f32>;

// // density field
// @group(0) @binding(6) var<storage, read_write> density: array<f32>;
// @group(0) @binding(7) var<storage, read_write> density_new: array<f32>;

// // divergence and pressure
// @group(0) @binding(8) var<storage, read_write> divergence: array<f32>;
// @group(0) @binding(9) var<storage, read_write> pressure: array<f32>;

// // density sources field
// @group(0) @binding(10) var<storage, read> density_sources: array<f32>;

// // velocity sources field
// @group(0) @binding(11) var<storage, read> u_sources: array<f32>;
// @group(0) @binding(12) var<storage, read> v_sources: array<f32>;


const densityAddSourcesPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "density_add_sources_main",
  },
});

const densityAddSourcesBindGroup = device.createBindGroup({
  layout: densityAddSourcesPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    // { binding: 2, resource: { buffer: uBuffer } },
    // { binding: 3, resource: { buffer: vBuffer } },
    // { binding: 4, resource: { buffer: newUBuffer } },
    // { binding: 5, resource: { buffer: newVBuffer } },
    { binding: 6, resource: { buffer: densityBuffer } },
    // { binding: 7, resource: { buffer: newDensityBuffer } },
    // { binding: 8, resource: { buffer: divergenceBuffer } },
    // { binding: 9, resource: { buffer: pressureBuffer } },
    { binding: 10, resource: { buffer: densitySourcesBuffer } },
    // { binding: 11, resource: { buffer: uSourcesBuffer } },
    // { binding: 12, resource: { buffer: vSourcesBuffer } },
  ],
});


const densityDiffusePipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "density_diffuse_main",
  },
});

const densityDiffuseBindGroupA = device.createBindGroup({
  layout: densityDiffusePipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    // { binding: 2, resource: { buffer: uBuffer } },
    // { binding: 3, resource: { buffer: vBuffer } },
    // { binding: 4, resource: { buffer: newUBuffer } },
    // { binding: 5, resource: { buffer: newVBuffer } },
    { binding: 6, resource: { buffer: densityBuffer } },
    { binding: 7, resource: { buffer: newDensityBuffer } },
    // { binding: 8, resource: { buffer: divergenceBuffer } },
    // { binding: 9, resource: { buffer: pressureBuffer } },
    // { binding: 10, resource: { buffer: densitySourcesBuffer } },
    // { binding: 11, resource: { buffer: uSourcesBuffer } },
    // { binding: 12, resource: { buffer: vSourcesBuffer } },
  ],
});

const densityDiffuseBindGroupB = device.createBindGroup({
  layout: densityDiffusePipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    // { binding: 2, resource: { buffer: uBuffer } },
    // { binding: 3, resource: { buffer: vBuffer } },
    // { binding: 4, resource: { buffer: newUBuffer } },
    // { binding: 5, resource: { buffer: newVBuffer } },
    { binding: 6, resource: { buffer: newDensityBuffer } },
    { binding: 7, resource: { buffer: densityBuffer } },
    // { binding: 8, resource: { buffer: divergenceBuffer } },
    // { binding: 9, resource: { buffer: pressureBuffer } },
    // { binding: 10, resource: { buffer: densitySourcesBuffer } },
    // { binding: 11, resource: { buffer: uSourcesBuffer } },
    // { binding: 12, resource: { buffer: vSourcesBuffer } },
  ],
});


const densityAdvectPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "density_advect_main",
  },
});

const densityAdvectBindGroup = device.createBindGroup({
  layout: densityAdvectPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: uBuffer } },
    { binding: 3, resource: { buffer: vBuffer } },
    // { binding: 4, resource: { buffer: newUBuffer } },
    // { binding: 5, resource: { buffer: newVBuffer } },
    { binding: 6, resource: { buffer: densityBuffer } },
    { binding: 7, resource: { buffer: newDensityBuffer } },
    // { binding: 8, resource: { buffer: divergenceBuffer } },
    // { binding: 9, resource: { buffer: pressureBuffer } },
    // { binding: 10, resource: { buffer: densitySourcesBuffer } },
    // { binding: 11, resource: { buffer: uSourcesBuffer } },
    // { binding: 12, resource: { buffer: vSourcesBuffer } },
  ],
});


const velocityAddSourcesPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "velocity_add_sources_main",
  },
});

const velocityAddSourcesBindGroup = device.createBindGroup({
  layout: velocityAddSourcesPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: uBuffer } },
    { binding: 3, resource: { buffer: vBuffer } },
    // { binding: 4, resource: { buffer: newUBuffer } },
    // { binding: 5, resource: { buffer: newVBuffer } },
    // { binding: 6, resource: { buffer: densityBuffer } },
    // { binding: 7, resource: { buffer: newDensityBuffer } },
    // { binding: 8, resource: { buffer: divergenceBuffer } },
    // { binding: 9, resource: { buffer: pressureBuffer } },
    // { binding: 10, resource: { buffer: densitySourcesBuffer } },
    { binding: 11, resource: { buffer: uSourcesBuffer } },
    { binding: 12, resource: { buffer: vSourcesBuffer } },
  ],
});


const velocityDiffusePipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "velocity_diffuse_main",
  },
});

const velocityDiffuseBindGroupA = device.createBindGroup({
  layout: velocityDiffusePipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: uBuffer } },
    { binding: 3, resource: { buffer: vBuffer } },
    { binding: 4, resource: { buffer: newUBuffer } },
    { binding: 5, resource: { buffer: newVBuffer } },
    // { binding: 6, resource: { buffer: densityBuffer } },
    // { binding: 7, resource: { buffer: newDensityBuffer } },
    // { binding: 8, resource: { buffer: divergenceBuffer } },
    // { binding: 9, resource: { buffer: pressureBuffer } },
    // { binding: 10, resource: { buffer: densitySourcesBuffer } },
    // { binding: 11, resource: { buffer: uSourcesBuffer } },
    // { binding: 12, resource: { buffer: vSourcesBuffer } },
  ],
});

const velocityDiffuseBindGroupB = device.createBindGroup({
  layout: velocityDiffusePipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: newUBuffer } },
    { binding: 3, resource: { buffer: newVBuffer } },
    { binding: 4, resource: { buffer: uBuffer } },
    { binding: 5, resource: { buffer: vBuffer } },
    // { binding: 6, resource: { buffer: densityBuffer } },
    // { binding: 7, resource: { buffer: newDensityBuffer } },
    // { binding: 8, resource: { buffer: divergenceBuffer } },
    // { binding: 9, resource: { buffer: pressureBuffer } },
    // { binding: 10, resource: { buffer: densitySourcesBuffer } },
    // { binding: 11, resource: { buffer: uSourcesBuffer } },
    // { binding: 12, resource: { buffer: vSourcesBuffer } },
  ],
});


const velocityDivergencePipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "velocity_divergence_main",
  },
});

const velocityDivergenceBindGroup = device.createBindGroup({
  layout: velocityDivergencePipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: uBuffer } },
    { binding: 3, resource: { buffer: vBuffer } },
    // { binding: 4, resource: { buffer: newUBuffer } },
    // { binding: 5, resource: { buffer: newVBuffer } },
    // { binding: 6, resource: { buffer: densityBuffer } },
    // { binding: 7, resource: { buffer: newDensityBuffer } },
    { binding: 8, resource: { buffer: divergenceBuffer } },
    { binding: 9, resource: { buffer: pressureBuffer } },
    // { binding: 10, resource: { buffer: densitySourcesBuffer } },
    // { binding: 11, resource: { buffer: uSourcesBuffer } },
    // { binding: 12, resource: { buffer: vSourcesBuffer } },
  ],
});


const velocityPressureSolvePipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "velocity_pressure_solve_main",
  },
});

const velocityPressureSolveBindGroup = device.createBindGroup({
  layout: velocityPressureSolvePipeline.getBindGroupLayout(0),
  entries: [
    // { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    // { binding: 2, resource: { buffer: uBuffer } },
    // { binding: 3, resource: { buffer: vBuffer } },
    // { binding: 4, resource: { buffer: newUBuffer } },
    // { binding: 5, resource: { buffer: newVBuffer } },
    // { binding: 6, resource: { buffer: densityBuffer } },
    // { binding: 7, resource: { buffer: newDensityBuffer } },
    { binding: 8, resource: { buffer: divergenceBuffer } },
    { binding: 9, resource: { buffer: pressureBuffer } },
    // { binding: 10, resource: { buffer: densitySourcesBuffer } },
    // { binding: 11, resource: { buffer: uSourcesBuffer } },
    // { binding: 12, resource: { buffer: vSourcesBuffer } },
  ],
});


const velocityProjectPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "velocity_project_main",
  },
});

const velocityProjectBindGroup = device.createBindGroup({
  layout: velocityProjectPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: uBuffer } },
    { binding: 3, resource: { buffer: vBuffer } },
    // { binding: 4, resource: { buffer: newUBuffer } },
    // { binding: 5, resource: { buffer: newVBuffer } },
    // { binding: 6, resource: { buffer: densityBuffer } },
    // { binding: 7, resource: { buffer: newDensityBuffer } },
    // { binding: 8, resource: { buffer: divergenceBuffer } },
    { binding: 9, resource: { buffer: pressureBuffer } },
    // { binding: 10, resource: { buffer: densitySourcesBuffer } },
    // { binding: 11, resource: { buffer: uSourcesBuffer } },
    // { binding: 12, resource: { buffer: vSourcesBuffer } },
  ],
});


const velocityAdvectPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "velocity_advect_main",
  },
});

const velocityAdvectBindGroup = device.createBindGroup({
  layout: velocityAdvectPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: uBuffer } },
    { binding: 3, resource: { buffer: vBuffer } },
    { binding: 4, resource: { buffer: newUBuffer } },
    { binding: 5, resource: { buffer: newVBuffer } },
    // { binding: 6, resource: { buffer: densityBuffer } },
    // { binding: 7, resource: { buffer: newDensityBuffer } },
    // { binding: 8, resource: { buffer: divergenceBuffer } },
    // { binding: 9, resource: { buffer: pressureBuffer } },
    // { binding: 10, resource: { buffer: densitySourcesBuffer } },
    // { binding: 11, resource: { buffer: uSourcesBuffer } },
    // { binding: 12, resource: { buffer: vSourcesBuffer } },
  ],
});


const setBoundaryPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "set_boundary_main",
  },
});

const setBoundaryBindGroup = device.createBindGroup({
  layout: setBoundaryPipeline.getBindGroupLayout(0),
  entries: [
    // { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: uBuffer } },
    { binding: 3, resource: { buffer: vBuffer } },
    { binding: 4, resource: { buffer: newUBuffer } },
    { binding: 5, resource: { buffer: newVBuffer } },
    { binding: 6, resource: { buffer: densityBuffer } },
    { binding: 7, resource: { buffer: newDensityBuffer } },
    { binding: 8, resource: { buffer: divergenceBuffer } },
    { binding: 9, resource: { buffer: pressureBuffer } },
    // { binding: 10, resource: { buffer: densitySourcesBuffer } },
    // { binding: 11, resource: { buffer: uSourcesBuffer } },
    // { binding: 12, resource: { buffer: vSourcesBuffer } },
  ],
});


const swapDensityPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "swap_density_main",
  },
});

const swapDensityBindGroup = device.createBindGroup({
  layout: swapDensityPipeline.getBindGroupLayout(0),
  entries: [
    // { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    // { binding: 2, resource: { buffer: uBuffer } },
    // { binding: 3, resource: { buffer: vBuffer } },
    // { binding: 4, resource: { buffer: newUBuffer } },
    // { binding: 5, resource: { buffer: newVBuffer } },
    { binding: 6, resource: { buffer: densityBuffer } },
    { binding: 7, resource: { buffer: newDensityBuffer } },
    // { binding: 8, resource: { buffer: divergenceBuffer } },
    // { binding: 9, resource: { buffer: pressureBuffer } },
    // { binding: 10, resource: { buffer: densitySourcesBuffer } },
    // { binding: 11, resource: { buffer: uSourcesBuffer } },
    // { binding: 12, resource: { buffer: vSourcesBuffer } },
  ],
});


const swapVelocityPipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: computeShaderModule,
    entryPoint: "swap_velocity_main",
  },
});

const swapVelocityBindGroup = device.createBindGroup({
  layout: swapVelocityPipeline.getBindGroupLayout(0),
  entries: [
    // { binding: 0, resource: { buffer: floatMetadataBuffer } },
    { binding: 1, resource: { buffer: uintMetadataBuffer } },
    { binding: 2, resource: { buffer: uBuffer } },
    { binding: 3, resource: { buffer: vBuffer } },
    { binding: 4, resource: { buffer: newUBuffer } },
    { binding: 5, resource: { buffer: newVBuffer } },
    // { binding: 6, resource: { buffer: densityBuffer } },
    // { binding: 7, resource: { buffer: newDensityBuffer } },
    // { binding: 8, resource: { buffer: divergenceBuffer } },
    // { binding: 9, resource: { buffer: pressureBuffer } },
    // { binding: 10, resource: { buffer: densitySourcesBuffer } },
    // { binding: 11, resource: { buffer: uSourcesBuffer } },
    // { binding: 12, resource: { buffer: vSourcesBuffer } },
  ],
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
    { binding: 0, resource: { buffer: uintMetadataBuffer } },
    { binding: 1, resource: { buffer: densityBuffer } },
  ],
});



// ----- Main simulation loop -----
const numJacobiIterations = 20;

const workgroupX = 16;
const workgroupY = 16;

const dispatchX = Math.ceil(width / workgroupX);
const dispatchY = Math.ceil(height / workgroupY);

function frame() {
  const commandEncoder = device.createCommandEncoder();

  // Compute pass
  {
    const computePass = commandEncoder.beginComputePass();

    // 1) Velocity step

    // 1.1) Add velocity sources
    computePass.setPipeline(velocityAddSourcesPipeline);
    computePass.setBindGroup(0, velocityAddSourcesBindGroup); // u,v (in place)
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    // 1.2) Diffuse velocity
    for (let i = 0; i < numJacobiIterations / 2; i++) {
      computePass.setPipeline(velocityDiffusePipeline);
      computePass.setBindGroup(0, velocityDiffuseBindGroupA); // u,v -> u_new,v_new
      computePass.dispatchWorkgroups(dispatchX, dispatchY);

      computePass.setPipeline(setBoundaryPipeline);
      computePass.setBindGroup(0, setBoundaryBindGroup);
      computePass.dispatchWorkgroups(dispatchX, dispatchY);

      computePass.setPipeline(velocityDiffusePipeline);
      computePass.setBindGroup(0, velocityDiffuseBindGroupB); // u_new,v_new -> u,v
      computePass.dispatchWorkgroups(dispatchX, dispatchY);

      computePass.setPipeline(setBoundaryPipeline);
      computePass.setBindGroup(0, setBoundaryBindGroup);
      computePass.dispatchWorkgroups(dispatchX, dispatchY);
    }

    // 1.3) Project velocity
    computePass.setPipeline(velocityDivergencePipeline);
    computePass.setBindGroup(0, velocityDivergenceBindGroup); // velocity, divergence (in place)
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    computePass.setPipeline(setBoundaryPipeline);
    computePass.setBindGroup(0, setBoundaryBindGroup);
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    for (let i = 0; i < numJacobiIterations; i++) {
      computePass.setPipeline(velocityPressureSolvePipeline);
      computePass.setBindGroup(0, velocityPressureSolveBindGroup); // pressure (in place)
      computePass.dispatchWorkgroups(dispatchX, dispatchY);

      computePass.setPipeline(setBoundaryPipeline);
      computePass.setBindGroup(0, setBoundaryBindGroup);
      computePass.dispatchWorkgroups(dispatchX, dispatchY);
    }

    computePass.setPipeline(velocityProjectPipeline);
    computePass.setBindGroup(0, velocityProjectBindGroup); // u,v (in place)
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    computePass.setPipeline(setBoundaryPipeline);
    computePass.setBindGroup(0, setBoundaryBindGroup);
    computePass.dispatchWorkgroups(dispatchX, dispatchY);


    // 1.4) Advect velocity
    computePass.setPipeline(velocityAdvectPipeline);
    computePass.setBindGroup(0, velocityAdvectBindGroup); // u,v -> u_new,v_new
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    computePass.setPipeline(setBoundaryPipeline);
    computePass.setBindGroup(0, setBoundaryBindGroup);
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    computePass.setPipeline(swapVelocityPipeline);
    computePass.setBindGroup(0, swapVelocityBindGroup); // u_new,v_new -> u,v
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    // 1.5) Project velocity again
    computePass.setPipeline(velocityDivergencePipeline);
    computePass.setBindGroup(0, velocityDivergenceBindGroup); // velocity, divergence (in place)
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    computePass.setPipeline(setBoundaryPipeline);
    computePass.setBindGroup(0, setBoundaryBindGroup);
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    for (let i = 0; i < numJacobiIterations; i++) {
      computePass.setPipeline(velocityPressureSolvePipeline);
      computePass.setBindGroup(0, velocityPressureSolveBindGroup); // pressure (in place)
      computePass.dispatchWorkgroups(dispatchX, dispatchY);

      computePass.setPipeline(setBoundaryPipeline);
      computePass.setBindGroup(0, setBoundaryBindGroup);
      computePass.dispatchWorkgroups(dispatchX, dispatchY);
    }

    computePass.setPipeline(velocityProjectPipeline);
    computePass.setBindGroup(0, velocityProjectBindGroup); // u,v (in place)
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    computePass.setPipeline(setBoundaryPipeline);
    computePass.setBindGroup(0, setBoundaryBindGroup);
    computePass.dispatchWorkgroups(dispatchX, dispatchY);


    // 2) Density step

    // 2.1) Add density sources
    computePass.setPipeline(densityAddSourcesPipeline);
    computePass.setBindGroup(0, densityAddSourcesBindGroup); // density (in place)
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    // 2.2) Diffuse density
    for (let i = 0; i < numJacobiIterations / 2; i++) {
      computePass.setPipeline(densityDiffusePipeline);
      computePass.setBindGroup(0, densityDiffuseBindGroupA);  // density -> density_new
      computePass.dispatchWorkgroups(dispatchX, dispatchY);

      computePass.setPipeline(setBoundaryPipeline);
      computePass.setBindGroup(0, setBoundaryBindGroup);
      computePass.dispatchWorkgroups(dispatchX, dispatchY);

      computePass.setPipeline(densityDiffusePipeline);
      computePass.setBindGroup(0, densityDiffuseBindGroupB); // density_new -> density
      computePass.dispatchWorkgroups(dispatchX, dispatchY);

      computePass.setPipeline(setBoundaryPipeline);
      computePass.setBindGroup(0, setBoundaryBindGroup);
      computePass.dispatchWorkgroups(dispatchX, dispatchY);
    }

    // 2.3) Advect density
    computePass.setPipeline(densityAdvectPipeline);
    computePass.setBindGroup(0, densityAdvectBindGroup); // density -> density_new
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    computePass.setPipeline(setBoundaryPipeline);
    computePass.setBindGroup(0, setBoundaryBindGroup);
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    computePass.setPipeline(swapDensityPipeline);;
    computePass.setBindGroup(0, swapDensityBindGroup); // density_new -> density
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