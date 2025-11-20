// Minimal compute + graphics demo (clean version)
// - compute: operates on a 1D storage buffer shaped (width * height) of f32 values
//   and shifts each pixel value one step to the right each frame (per row)
// - graphics: samples that buffer and renders it as a grayscale image on the canvas

// High-level design:
// - Two storage buffers of f32 (ping-pong): bufferA, bufferB.
//   * Each element represents one pixel, stored row-major: index = y * width + x.
//   * At start, bufferA is seeded: leftmost column is 1.0, others 0.0.
// - Compute shader per frame:
//   * Reads from one buffer and writes into the other.
//   * For each (x, y), it sets out(x, y) = in(x-1, y), with x=0 keeping its original seed value.
// - Graphics shader per frame:
//   * Reads from the CURRENT buffer (the one we just wrote) as a storage buffer.
//   * For each fragment (pixel), compute its (x, y), look up the scalar, and output vec4(val, val, val, 1).

// This keeps everything in buffers (no textures in compute), and uses a storage buffer
// directly in the fragment shader to visualize the data. Bindings are consistent and simple.

const adapter = (await navigator.gpu.requestAdapter())!;
const device = await adapter.requestDevice();

const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
const context = canvas.getContext("webgpu") as GPUCanvasContext;
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format: canvasFormat, alphaMode: "premultiplied" });

// -------------------------------
// Canvas + data dimensions
// -------------------------------
const dpr = window.devicePixelRatio || 1;
const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
const pixelCount = width * height;
console.log("canvas size (device px):", width, height, "dpr:", dpr);

// -------------------------------
// Seed data: one f32 per pixel, row-major y*width + x
// Leftmost column = 1.0, all other entries = 0.0
// -------------------------------
const seed = new Float32Array(pixelCount);
for (let y = 0; y < height; ++y) {
  seed[y * width + 0] = 1.0;
}

// -------------------------------
// GPU buffers
// -------------------------------
// Ping-pong storage buffers (A and B). We'll alternate which is src/dst each frame.
const bufferSizeBytes = seed.byteLength;
const bufferA = device.createBuffer({
  size: bufferSizeBytes,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const bufferB = device.createBuffer({
  size: bufferSizeBytes,
  usage: GPUBufferUsage.STORAGE,
});
// Upload seed data into bufferA
device.queue.writeBuffer(bufferA, 0, seed.buffer, seed.byteOffset, seed.byteLength);

// Uniform buffer for size (width, height) as u32
const sizeU32 = new Uint32Array([width, height]);
const sizeBuffer = device.createBuffer({
  size: sizeU32.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(sizeBuffer, 0, sizeU32.buffer, sizeU32.byteOffset, sizeU32.byteLength);

// -------------------------------
// WGSL: compute shader
// @group(0) @binding(0): read-only storage buffer (src buffer)
// @group(0) @binding(1): write-only storage buffer (dst buffer)
// @group(0) @binding(2): uniform vec2<u32> for (width, height)
// For each pixel (x, y):
//   if x == 0: keep original seed value (from src)
//   else: dst(x, y) = src(x - 1, y)
// -------------------------------
const computeShaderCode = `
struct SizeInfo {
  width  : u32,
  height : u32,
};

@group(0) @binding(0) var<storage, read> srcBuf : array<f32>;
@group(0) @binding(1) var<storage, read_write> dstBuf : array<f32>;
@group(0) @binding(2) var<uniform> uSize : SizeInfo;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let x : u32 = gid.x;
  let y : u32 = gid.y;

  if (x >= uSize.width || y >= uSize.height) {
    return;
  }

  let idx : u32 = y * uSize.width + x;
  var val : f32;

  if (x == 0u) {
    // leftmost column: loop around to rightmost column
    val = srcBuf[y * uSize.width + (uSize.width - 1u)];
  } else {
    // shift one pixel to the right: copy from (x-1, y)
    let leftIdx : u32 = y * uSize.width + (x - 1u);
    val = srcBuf[leftIdx];
  }

  dstBuf[idx] = val;
}
`;

// -------------------------------
// WGSL: graphics shader
// @group(0) @binding(0): read-only storage buffer (current buffer)
// @group(0) @binding(1): uniform vec2<u32> for (width, height)
// Vertex: fullscreen triangle with UV from clip space
// Fragment: compute (x, y) from UV, read corresponding value, output grayscale
// -------------------------------
const graphicsShaderCode = `
struct SizeInfo {
  width  : u32,
  height : u32,
};

@group(0) @binding(0) var<storage, read> dataBuf : array<f32>;
@group(0) @binding(1) var<uniform> uSize : SizeInfo;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vi : u32) -> VSOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );

  var out : VSOut;
  let pos = positions[vi];
  out.pos = vec4<f32>(pos, 0.0, 1.0);
  out.uv = pos * 0.5 + vec2<f32>(0.5, 0.5);
  return out;
}

@fragment
fn fragment_main(in : VSOut) -> @location(0) vec4<f32> {
  // Convert UV back to pixel coords (clamp to edge)
  let px = clamp(i32(in.uv.x * f32(uSize.width)),  0, i32(uSize.width)  - 1);
  let py = clamp(i32(in.uv.y * f32(uSize.height)), 0, i32(uSize.height) - 1);

  let idx : u32 = u32(py) * uSize.width + u32(px);
  let v : f32 = dataBuf[idx];

  return vec4<f32>(v, v, v, 1.0);
}
`;

// -------------------------------
// Create shader modules
// -------------------------------
const computeModule = device.createShaderModule({ code: computeShaderCode });
const graphicsModule = device.createShaderModule({ code: graphicsShaderCode });

// -------------------------------
// Compute pipeline + bind groups (ping-pong)
// -------------------------------
const computeBindGroupLayout = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
  ],
});

const computePipeline = device.createComputePipeline({
  layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
  compute: { module: computeModule, entryPoint: "main" },
});

// Bind groups: A -> B and B -> A
const computeBindGroupAB = device.createBindGroup({
  layout: computeBindGroupLayout,
  entries: [
    { binding: 0, resource: { buffer: bufferA } },
    { binding: 1, resource: { buffer: bufferB } },
    { binding: 2, resource: { buffer: sizeBuffer } },
  ],
});

const computeBindGroupBA = device.createBindGroup({
  layout: computeBindGroupLayout,
  entries: [
    { binding: 0, resource: { buffer: bufferB } },
    { binding: 1, resource: { buffer: bufferA } },
    { binding: 2, resource: { buffer: sizeBuffer } },
  ],
});

// -------------------------------
// Graphics pipeline + bind groups (also swap which buffer we visualize)
// -------------------------------
const graphicsBindGroupLayout = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
  ],
});

const renderPipeline = device.createRenderPipeline({
  layout: device.createPipelineLayout({ bindGroupLayouts: [graphicsBindGroupLayout] }),
  vertex: { module: graphicsModule, entryPoint: "vertex_main", buffers: [] },
  fragment: { module: graphicsModule, entryPoint: "fragment_main", targets: [{ format: canvasFormat }] },
  primitive: { topology: "triangle-list" },
});

const graphicsBindGroupA = device.createBindGroup({
  layout: graphicsBindGroupLayout,
  entries: [
    { binding: 0, resource: { buffer: bufferA } },
    { binding: 1, resource: { buffer: sizeBuffer } },
  ],
});

const graphicsBindGroupB = device.createBindGroup({
  layout: graphicsBindGroupLayout,
  entries: [
    { binding: 0, resource: { buffer: bufferB } },
    { binding: 1, resource: { buffer: sizeBuffer } },
  ],
});

// -------------------------------
// Animation loop: compute then render each frame
//   - Ping-pong buffers so the shift accumulates over time.
// -------------------------------
let ping = true; // true: A->B and show B, false: B->A and show A

const workgroupX = 16;
const workgroupY = 16;
const dispatchX = Math.ceil(width / workgroupX);
const dispatchY = Math.ceil(height / workgroupY);

function frame() {
  const encoder = device.createCommandEncoder();

  // Compute pass: shift values right, ping-pong between buffers
  {
    const pass = encoder.beginComputePass();
    pass.setPipeline(computePipeline);
    pass.setBindGroup(0, ping ? computeBindGroupAB : computeBindGroupBA);
    pass.dispatchWorkgroups(dispatchX, dispatchY);
    pass.end();
  }

  // Render pass: visualize the CURRENT buffer
  {
    const view = context.getCurrentTexture().createView();
    const rpd: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view,
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        },
      ],
    };

    const rpass = encoder.beginRenderPass(rpd);
    rpass.setPipeline(renderPipeline);
    rpass.setBindGroup(0, ping ? graphicsBindGroupB : graphicsBindGroupA);
    rpass.draw(3);
    rpass.end();
  }

  device.queue.submit([encoder.finish()]);

  // Swap roles for next frame
  ping = !ping;
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);

console.log("index.ts: compute+graphics shift-right demo running.");
