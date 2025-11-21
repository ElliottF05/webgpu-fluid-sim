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
@group(0) @binding(6) var<storage, read> pressure: array<f32>;
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

fn is_obstacle(i: u32, j: u32) -> bool {
    return obstacles[idx_center(i, j)] > 0.5;
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


@compute @workgroup_size(16, 16)
fn divergence_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    // this assumes cell_size = 1.0 for simplicity
    let i = global_id.x;
    let j = global_id.y;

    if (i >= uint_metadata.width || j >= uint_metadata.height) {
        return;
    }

    let idx = idx_center(i, j);
    if (is_obstacle(i, j)) {
        divergence[idx] = 0.0;
        return;
    }

    var u_right = 0.0;
    var u_left = 0.0;
    var v_top = 0.0;
    var v_bottom = 0.0;

    if (i + 1u < uint_metadata.width && !is_obstacle(i + 1u, j)) {
        u_right = u[idx_u(i + 1u, j)];
    }
    if (i > 0u && !is_obstacle(i - 1u, j)) {
        u_left = u[idx_u(i, j)];
    }
    if (j + 1u < uint_metadata.height && !is_obstacle(i, j + 1u)) {
        v_top = v[idx_v(i, j + 1u)];
    }
    if (j > 0u && !is_obstacle(i, j - 1u)) {
        v_bottom = v[idx_v(i, j)];
    }

    // let u_right = u[idx_u(i + 1u, j)];
    // let u_left = u[idx_u(i, j)];
    // let v_top = v[idx_v(i, j + 1u)];
    // let v_bottom = v[idx_v(i, j)];

    // TODO: might need to normalize (divide) by cell size here
    let div = (u_right - u_left) + (v_top - v_bottom);
    divergence[idx_center(i, j)] = div;
}



@compute @workgroup_size(16, 16)
fn pressure_solve_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    // this assumes cell_size = 1.0 for simplicity
    let i = global_id.x;
    let j = global_id.y;

    if (i >= uint_metadata.width || j >= uint_metadata.height) {
        return;
    }

    let idx = idx_center(i, j);
    if (is_obstacle(i, j)) {
        new_pressure[idx] = 0.0;
        return;
    }

    var num_neighbors = 0.0;
    var sum_neighbors = 0.0;

    if (i > 0u && !is_obstacle(i - 1u, j)) {
        sum_neighbors = sum_neighbors + pressure[idx_center(i - 1u, j)];
        num_neighbors = num_neighbors + 1.0;
    }
    if (i + 1u < uint_metadata.width && !is_obstacle(i + 1u, j)) {
        sum_neighbors = sum_neighbors + pressure[idx_center(i + 1u, j)];
        num_neighbors = num_neighbors + 1.0;
    }
    if (j > 0u && !is_obstacle(i, j - 1u)) {
        sum_neighbors = sum_neighbors + pressure[idx_center(i, j - 1u)];
        num_neighbors = num_neighbors + 1.0;
    }
    if (j + 1u < uint_metadata.height && !is_obstacle(i, j + 1u)) {
        sum_neighbors = sum_neighbors + pressure[idx_center(i, j + 1u)];
        num_neighbors = num_neighbors + 1.0;
    }

    let div = divergence[idx_center(i, j)];
    // let new_p = (div + sum_neighbors) / num_neighbors;
    let new_p = (sum_neighbors - div) / num_neighbors;

    new_pressure[idx_center(i, j)] = new_p;
}


@compute @workgroup_size(16, 16)
fn project_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    // this assumes cell_size = 1.0 for simplicity
    let i = global_id.x;
    let j = global_id.y;


    // project u
    // u is a grid of size (width + 1) x height
    if (i > 0u && i < uint_metadata.width && j < uint_metadata.height) {
        let idx_left = idx_center(i - 1u, j);
        let idx_right = idx_center(i, j);

        let solid_left = is_obstacle(i - 1u, j);
        let solid_right = is_obstacle(i, j);

        let face_idx = idx_u(i, j);

        if (solid_left || solid_right) {
            u_new[face_idx] = 0.0;
            return;
        } else {
            let p_dif = pressure[idx_right] - pressure[idx_left];
            u_new[face_idx] = u[face_idx] - p_dif;
        }
    }

    // project v
    // v is a grid of size width x (height + 1)
    if (i < uint_metadata.width && j > 0u && j < uint_metadata.height) {
        let idx_bottom = idx_center(i, j - 1u);
        let idx_top = idx_center(i, j);

        let solid_bottom = is_obstacle(i, j - 1u);
        let solid_top = is_obstacle(i, j);

        let face_idx = idx_v(i, j);
        
        if (solid_bottom || solid_top) {
            v_new[face_idx] = 0.0;
            return;
        } else {
            let p_dif = pressure[idx_top] - pressure[idx_bottom];
            v_new[face_idx] = v[face_idx] - p_dif;
        }
    }
}


@compute @workgroup_size(16, 16)
fn cleanup_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    let i = global_id.x;
    let j = global_id.y;

    // swap dye and dye_new
    if (i < uint_metadata.width && j < uint_metadata.height) {
        dye_new[idx_center(i, j)] = dye[idx_center(i, j)];
    }
}