struct FloatMetadata {
    delta_time: f32,
    over_relaxation: f32,
}

struct UintMetadata {
    width: u32,
    height: u32,
    num_iters: u32,
}

// input and output buffers
@group(0) @binding(0) var<uniform> float_metadata: FloatMetadata;
@group(0) @binding(1) var<uniform> uint_metadata: UintMetadata;

// velocity fields (staggered grid)
@group(0) @binding(2) var<storage, read> u: array<f32>; // size = (width + 1) * height
@group(0) @binding(3) var<storage, read> v: array<f32>; // size = width * (height + 1)
@group(0) @binding(4) var<storage, read_write> u_new: array<f32>;
@group(0) @binding(5) var<storage, read_write> v_new: array<f32>;

// these values are at cell centers, so will have size width x height
@group(0) @binding(6) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(7) var<storage, read_write> new_pressure: array<f32>;
@group(0) @binding(8) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(9) var<storage, read> obstacles: array<f32>;
@group(0) @binding(10) var<storage, read> dye: array<f32>;
@group(0) @binding(11) var<storage, read_write> dye_new: array<f32>;


// helper functions
fn idx_u(i: u32, j: u32) -> u32 {
    return j * (uint_metadata.width + 1u) + i;
}

fn idx_v(i: u32, j: u32) -> u32 {
    return j * uint_metadata.width + i;
}

fn idx_center(i: u32, j: u32) -> u32 {
    return j * uint_metadata.width + i;
}

fn sample_u(x: f32, y: f32) -> f32 {
    // each u value is at (i, j + 0.5)
    // clamp position to u's valid domain
    let x_c = clamp(x, 0.0, f32(uint_metadata.width));
    let y_c = clamp(y, 0.5, f32(uint_metadata.height) - 0.5);
    
    let i = u32(floor(x_c));
    let j = u32(floor(y_c - 0.5));
    let s = x_c - f32(i);
    let t = y_c - f32(j) - 0.5;

    let u00 = u[idx_u(i, j)];
    let u10 = u[idx_u(i + 1u, j)];
    let u01 = u[idx_u(i, j + 1u)];
    let u11 = u[idx_u(i + 1u, j + 1u)];

    return  (1.0 - s) * (1.0 - t) * u00 +
                    s * (1.0 - t) * u10 +
            (1.0 - s) *         t * u01 +
                    s *         t * u11;
}

fn sample_v(x: f32, y: f32) -> f32 {
    // each v value is at (i + 0.5, j)
    // clamp position to v's valid domain
    let x_c = clamp(x, 0.5, f32(uint_metadata.width) - 0.5);
    let y_c = clamp(y, 0.0, f32(uint_metadata.height));

    let i = u32(floor(x_c - 0.5));
    let j = u32(floor(y_c));
    let s = x_c - f32(i) - 0.5;
    let t = y_c - f32(j);

    let v00 = v[idx_v(i, j)];
    let v10 = v[idx_v(i + 1u, j)];
    let v01 = v[idx_v(i, j + 1u)];
    let v11 = v[idx_v(i + 1u, j + 1u)];

    return  (1.0 - s) * (1.0 - t) * v00 +
                    s * (1.0 - t) * v10 +
            (1.0 - s) *         t * v01 +
                    s *         t * v11;
}

fn sample_pressure(x: f32, y: f32) -> f32 {
    // each pressure value is at (i + 0.5, j + 0.5)
    // clamp position to pressure's valid domain
    let x_c = clamp(x, 0.5, f32(uint_metadata.width) - 0.5);
    let y_c = clamp(y, 0.5, f32(uint_metadata.height) - 0.5);

    let i = u32(floor(x_c - 0.5));
    let j = u32(floor(y_c - 0.5));
    let s = x_c - f32(i) - 0.5;
    let t = y_c - f32(j) - 0.5;

    let c00 = pressure[idx_center(i, j)];
    let c10 = pressure[idx_center(i + 1u, j)];
    let c01 = pressure[idx_center(i, j + 1u)];
    let c11 = pressure[idx_center(i + 1u, j + 1u)];

    return  (1.0 - s) * (1.0 - t) * c00 +
                    s * (1.0 - t) * c10 +
            (1.0 - s) *         t * c01 +
                    s *         t * c11;
}

fn sample_dye(x: f32, y: f32) -> f32 {
    // each dye value is at (i + 0.5, j + 0.5)
    // clamp position to dye's valid domain
    let x_c = clamp(x, 0.5, f32(uint_metadata.width) - 0.5);
    let y_c = clamp(y, 0.5, f32(uint_metadata.height) - 0.5); 
    
    let i = u32(floor(x_c - 0.5));
    let j = u32(floor(y_c - 0.5));
    let s = x_c - f32(i) - 0.5;
    let t = y_c - f32(j) - 0.5;

    let c00 = dye[idx_center(i, j)];
    let c10 = dye[idx_center(i + 1u, j)];
    let c01 = dye[idx_center(i, j + 1u)];
    let c11 = dye[idx_center(i + 1u, j + 1u)];

    return  (1.0 - s) * (1.0 - t) * c00 +
                    s * (1.0 - t) * c10 +
            (1.0 - s) *         t * c01 +
                    s *         t * c11;
}

fn sample_divergence(x: f32, y: f32) -> f32 {
    // each divergence value is at (i + 0.5, j + 0.5)
    // clamp position to dye's valid domain
    let x_c = clamp(x, 0.5, f32(uint_metadata.width) - 0.5);
    let y_c = clamp(y, 0.5, f32(uint_metadata.height) - 0.5); 
    
    let i = u32(floor(x_c - 0.5));
    let j = u32(floor(y_c - 0.5));
    let s = x_c - f32(i) - 0.5;
    let t = y_c - f32(j) - 0.5;

    let c00 = divergence[idx_center(i, j)];
    let c10 = divergence[idx_center(i + 1u, j)];
    let c01 = divergence[idx_center(i, j + 1u)];
    let c11 = divergence[idx_center(i + 1u, j + 1u)];

    return  (1.0 - s) * (1.0 - t) * c00 +
                    s * (1.0 - t) * c10 +
            (1.0 - s) *         t * c01 +
                    s *         t * c11;
}



@compute @workgroup_size(16, 16)
fn advect_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    // use backwards semi-lagrangian advection
    
    // this assumes cell_size = 1.0 for simplicity
    let i = global_id.x;
    let j = global_id.y;

    if (i >= uint_metadata.width || j >= uint_metadata.height) {
        return;
    }


    // advect u
    // first get velocity at the u location
    {
        let x = f32(i);
        let y = f32(j) + 0.5;
        if (x > 0.0 && x < f32(uint_metadata.width) && y > 0.0 && y < f32(uint_metadata.height)) {
            let vel_x = u[idx_u(i, j)];
            let vel_y = sample_v(x, y);

            let x_prev = x - float_metadata.delta_time * vel_x;
            let y_prev = y - float_metadata.delta_time * vel_y;

            u_new[idx_u(i, j)] = sample_u(x_prev, y_prev);
        }
    }


    // advect v
    // first get velocity at the v location
    {
        let x = f32(i) + 0.5;
        let y = f32(j);
        if (x > 0.0 && x < f32(uint_metadata.width) && y > 0.0 && y < f32(uint_metadata.height)) {
            let vel_x = sample_u(x, y);
            let vel_y = v[idx_v(i, j)];

            let x_prev = x - float_metadata.delta_time * vel_x;
            let y_prev = y - float_metadata.delta_time * vel_y;

            v_new[idx_v(i, j)] = sample_v(x_prev, y_prev);
        }
    }

    // advect dye
    // first get velocity at the cell center
    {
        let x = f32(i) + 0.5;
        let y = f32(j) + 0.5;
        if (x > 0.0 && x < f32(uint_metadata.width) && y > 0.0 && y < f32(uint_metadata.height)) {
            let vel_x = sample_u(x, y);
            let vel_y = sample_v(x, y);

            let x_prev = x - float_metadata.delta_time * vel_x;
            let y_prev = y - float_metadata.delta_time * vel_y;

            dye_new[idx_center(i, j)] = sample_dye(x_prev, y_prev);
        }
    }
}

