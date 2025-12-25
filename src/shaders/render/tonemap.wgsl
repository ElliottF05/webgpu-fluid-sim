// STRUCTS

struct FloatMetadata {
    grav_constant: f32,
    delta_time: f32,
    epsilon_multiplier: f32,
    bh_theta: f32,
    cam_center: vec2<f32>,
    cam_half_size: vec2<f32>,
    viewport: vec2<f32>,
}

struct UintMetadata {
    num_bodies: u32,
}

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}


// BINDINGS AND BUFFERS

// metadata buffers
@group(0) @binding(0) var<uniform> float_metadata: FloatMetadata;
@group(0) @binding(1) var<uniform> uint_metadata: UintMetadata;


// data buffers
@group(0) @binding(2) var density_tex: texture_2d<f32>;

// sampler
@group(0) @binding(3) var density_sampler: sampler;


// HELPER FUNCTIONS

fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn colormap(t: f32) -> vec3<f32> {
    let x = saturate(t);

    // gradient colors
    let c0 = vec3<f32>(0.0, 0.0, 0.0);
    let c1 = vec3<f32>(0.10, 0.20, 0.90);
    let c2 = vec3<f32>(0.20, 0.90, 0.90);
    let c3 = vec3<f32>(1.0, 1.0, 1.0);

    // assigning to gradient segments
    if x < 0.33 {
        let a = x / 0.33;
        return mix(c0, c1, a);
    } else if x < 0.66 {
        let a = (x - 0.33) / 0.33;
        return mix(c1, c2, a);
    } else {
        let a = (x - 0.66) / 0.34;
        return mix(c2, c3, a);
    }
}


// MAIN SHADERS

@vertex
fn vertex_main(@builtin(vertex_index) vid: u32) -> VSOut {
    // uses a single fullscreen triangle
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );

    var out: VSOut;
    out.pos = vec4<f32>(pos[vid], 0.0, 1.0);

    // map from ndc [-1,1] to uv [0,1]
    out.uv = out.pos.xy * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fragment_main(in: VSOut) -> @location(0) vec4<f32> {
    // sample density texture
    let density = textureSample(density_tex, density_sampler, in.uv).r;

    let exposure = 8.0; // change this
    let x = exposure * density;

    // tone mapping (Reinhard)
    let mapped = x / (1.0 + x);

    // gamma correction
    let gamma = 1.0 / 2.2;
    let mapped_gamma = pow(clamp(mapped, 0.0, 1.0), gamma);

    // add color
    let color = colormap(mapped_gamma);

    return vec4<f32>(color, 1.0);
}