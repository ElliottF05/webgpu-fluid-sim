// STRUCTS

struct Metadata {
    cam_center: vec2<f32>,
    cam_half_size: vec2<f32>,
    viewport: vec2<f32>,
    num_bodies: u32,
    _pad0: u32,
}

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>, // quad local coords in [-1,1]
    @location(1) @interpolate(flat) radius_px: f32,
};


// BINDINGS AND BUFFERS

// metadata buffers
@group(0) @binding(0) var<uniform> metadata: Metadata;

// data buffers
@group(0) @binding(1) var<storage, read> pos: array<vec2<f32>>;


// HELPER FUNCTIONS

fn world_to_ndc(p: vec2<f32>) -> vec2<f32> {
    // maps to [-1,1]
    return (p - metadata.cam_center) / metadata.cam_half_size;
}


// MAIN SHADERS

@vertex
fn vertex_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VSOut {
    let quad = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );

    var out: VSOut;

    if iid >= metadata.num_bodies {
        out.pos = vec4<f32>(2.0, 2.0, 0.0, 1.0); // offscreen
        out.uv = vec2<f32>(0.0);
        out.radius_px = 0.0;
        return out;
    }

    let world_pos = pos[iid];
    let center_ndc = world_to_ndc(world_pos);

    var radius_px = 2.0;
    let px_to_ndc = 2.0 / metadata.viewport;
    let radius_ndc = radius_px * px_to_ndc;

    let uv = quad[vid];
    out.pos = vec4<f32>(center_ndc + uv * radius_ndc, 0.0, 1.0);
    out.uv = uv;
    out.radius_px = radius_px;

    return out;
}

@fragment
fn fragment_main(in: VSOut) -> @location(0) vec4<f32> {
    // soft kernel (gaussian-like)
    let r = length(in.uv);

    if r > 1.0 {
        discard;
    }

    let k = 4.0; // higher = sharper
    let w = exp(-k * r * r);

    // compensate for zoom by reducing per-pixel contribution when zoomed out
    let zoom = max(metadata.cam_half_size.x, metadata.cam_half_size.y);
    let zoom_scale = 1.0 / (1.0 + 0.05 * zoom); // tweak 0.05 to adjust falloff speed

    let base = 0.02; // change this to adjust overall brightness

    let density = base * w * zoom_scale;

    return vec4<f32>(density, 0.0, 0.0, 1.0);
}