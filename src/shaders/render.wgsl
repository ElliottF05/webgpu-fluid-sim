// STRUCTS

struct FloatMetadata {
    grav_constant: f32,
    delta_time: f32,
    epsilon_multiplier: f32,
    _pad: f32,
    cam_center: vec2<f32>,
    cam_half_size: vec2<f32>,
    viewport: vec2<f32>,
}

struct UintMetadata {
    num_bodies: u32,
}

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>, // quad local coords in [-1,1]
    @location(1) @interpolate(flat) is_user_body: u32,
};


// BINDINGS AND BUFFERS

// metadata buffers
@group(0) @binding(0) var<uniform> float_metadata: FloatMetadata;
@group(0) @binding(1) var<uniform> uint_metadata: UintMetadata;

// data buffers
@group(0) @binding(2) var<storage, read> pos_buf: array<vec2<f32>>;


// HELPER FUNCTIONS

fn world_to_ndc(p: vec2<f32>) -> vec2<f32> {
    // maps to [-1,1]
    return (p - float_metadata.cam_center) / float_metadata.cam_half_size;
}


// MAIN SHADERS

@vertex
fn vertex_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VSOut {
    // 6 vertices = 2 triangles making a quad
    let quad = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );

    var out: VSOut;

    out.is_user_body = select(0u, 1u, iid == 0u);

    if iid >= uint_metadata.num_bodies {
        out.pos = vec4<f32>(2.0, 2.0, 0.0, 1.0); // off-screen
        out.uv = vec2<f32>(0.0);
        return out;
    }

    let world_pos = pos_buf[iid];
    let center_ndc = world_to_ndc(world_pos);

    // pixel and NDC radius
    var radius_px: f32 = 1.0;
    if iid == 0u {
        // make user-controlled body slightly larger
        radius_px = 4.0;
    }
    let px_to_ndc = 2.0 / float_metadata.viewport;
    let radius_ndc = radius_px * px_to_ndc;

    let uv = quad[vid];
    let pos_ndc = center_ndc + uv * radius_ndc;

    out.pos = vec4<f32>(pos_ndc, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fragment_main(in: VSOut) -> @location(0) vec4<f32> {
    // round mask
    let r = length(in.uv);

    // anti aliasing
    let aa_width = max(fwidth(r), 0.01);
    let edge = 1.0 - smoothstep(1.0 - aa_width, 1.0 + aa_width, r);

    let alpha = edge;

    // base color
    let color = vec3<f32>(0.85, 0.88, 0.95);
    if in.is_user_body == 1u {
        // user-controlled body color
        return vec4<f32>(0.9, 0.4, 0.3, alpha);
    }

    return vec4<f32>(0.3 * color, alpha);
}