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


// BINDINGS AND BUFFERS

// metadata buffers
@group(0) @binding(0) var<uniform> float_metadata: FloatMetadata;
@group(0) @binding(1) var<uniform> uint_metadata: UintMetadata;