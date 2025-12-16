struct FloatMetadata {
    grav_constant: f32,
    delta_time: f32,
    cam_center: vec2<f32>,
    cam_half_size: vec2<f32>,
    viewport: vec2<f32>,
}

struct UintMetadata {
    num_bodies: u32,
}

// metadata buffers
@group(0) @binding(0) var<uniform> float_metadata: FloatMetadata;
@group(0) @binding(1) var<uniform> uint_metadata: UintMetadata;

// data buffers
@group(0) @binding(2) var<storage, read_write> mass_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> pos_buf: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> vel_buf: array<vec2<f32>>;


// USING VELOCITY VERLET INTEGRATION

@compute @workgroup_size(16)
fn half_vel_step_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= uint_metadata.num_bodies {
        return;
    }

    let pos1 = pos_buf[i];
    let m1 = mass_buf[i];
    var f_net = vec2<f32>(0.0, 0.0);

    // accumulate force from all other bodies
    for (var j: u32 = 0; j < uint_metadata.num_bodies; j++) {
        if j == i {
            continue;
        }
        let pos2 = pos_buf[j];
        let m2 = mass_buf[j];

        let r = pos2 - pos1;
        let dist = length(r);

        let f_grav = float_metadata.grav_constant * m1 * m2 * r / (dist * dist * dist);
        f_net += f_grav;
    }

    // update velocity by a half step
    let accel = f_net / m1;
    let dt = float_metadata.delta_time;
    vel_buf[i] = vel_buf[i] + 0.5 * accel * dt;
}

@compute @workgroup_size(16)
fn pos_step_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= uint_metadata.num_bodies {
        return;
    }

    // update position using half velocity
    let dt = float_metadata.delta_time;
    pos_buf[i] = pos_buf[i] + vel_buf[i] * dt;
}