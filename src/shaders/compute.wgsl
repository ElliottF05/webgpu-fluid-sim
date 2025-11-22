struct FloatMetadata {
    delta_time: f32,
    cell_size: f32,
    diffusion_rate: f32,
    viscosity: f32,
}

struct UintMetadata {
    width: u32,
    height: u32,
}

// metadata buffers
@group(0) @binding(0) var<uniform> float_metadata: FloatMetadata;
@group(0) @binding(1) var<uniform> uint_metadata: UintMetadata;

// velocity fields
@group(0) @binding(2) var<storage, read_write> u: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> u_new: array<f32>;
@group(0) @binding(5) var<storage, read_write> v_new: array<f32>;

// density field
@group(0) @binding(6) var<storage, read_write> density: array<f32>;
@group(0) @binding(7) var<storage, read_write> density_new: array<f32>;

// divergence and pressure
@group(0) @binding(8) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(9) var<storage, read_write> pressure: array<f32>;

// density sources field
@group(0) @binding(10) var<storage, read> density_sources: array<f32>;
@group(0) @binding(11) var<storage, read> density_constants: array<f32>;

// velocity sources field
@group(0) @binding(12) var<storage, read> u_sources: array<f32>;
@group(0) @binding(13) var<storage, read> v_sources: array<f32>;
@group(0) @binding(14) var<storage, read> u_constants: array<f32>;
@group(0) @binding(15) var<storage, read> v_constants: array<f32>;

// boundary type field
@group(0) @binding(16) var<storage, read> obstacles: array<u32>;




// helper functions
fn idx_center(i: u32, j: u32) -> u32 {
    return j * uint_metadata.width + i;
}

fn sample_density(x: f32, y: f32) -> f32 {
    // each density value is centered at (i,j)
    // use interpolation
    let i = u32(floor(x));
    let j = u32(floor(y));
    
    let s = x - f32(i);
    let t = y - f32(j);

    let d00 = density[idx_center(i, j)];
    let d10 = density[idx_center(i + 1u, j)];
    let d01 = density[idx_center(i, j + 1u)];
    let d11 = density[idx_center(i + 1u, j + 1u)];

    return  (1.0 - s) * (1.0 - t) * d00 +
                    s * (1.0 - t) * d10 +
            (1.0 - s) *         t * d01 +
                    s *         t * d11;
}

fn sample_u(x: f32, y: f32) -> f32 {
    // each density value is centered at (i,j)
    // use interpolation
    let i = u32(floor(x));
    let j = u32(floor(y));
    
    let s = x - f32(i);
    let t = y - f32(j);

    let u00 = u[idx_center(i, j)];
    let u10 = u[idx_center(i + 1u, j)];
    let u01 = u[idx_center(i, j + 1u)];
    let u11 = u[idx_center(i + 1u, j + 1u)];

    return  (1.0 - s) * (1.0 - t) * u00 +
                    s * (1.0 - t) * u10 +
            (1.0 - s) *         t * u01 +
                    s *         t * u11;
}

fn sample_v(x: f32, y: f32) -> f32 {
    // each density value is centered at (i,j)
    // use interpolation
    let i = u32(floor(x));
    let j = u32(floor(y));
    
    let s = x - f32(i);
    let t = y - f32(j);

    let v00 = v[idx_center(i, j)];
    let v10 = v[idx_center(i + 1u, j)];
    let v01 = v[idx_center(i, j + 1u)];
    let v11 = v[idx_center(i + 1u, j + 1u)];

    return  (1.0 - s) * (1.0 - t) * v00 +
                    s * (1.0 - t) * v10 +
            (1.0 - s) *         t * v01 +
                    s *         t * v11;
}


@compute @workgroup_size(16, 16)
fn density_add_sources_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= uint_metadata.width || j >= uint_metadata.height) {
        return;
    }

    let idx = idx_center(i, j);

    if (density_constants[idx] != 0.0) {
        density[idx] = density_constants[idx];
        return;
    }
    
    density[idx] += float_metadata.delta_time * density_sources[idx];
}


@compute @workgroup_size(16, 16)
fn density_diffuse_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    let i = global_id.x;
    let j = global_id.y;

    // don't process edge cells
    if (i == 0u || j == 0u || i >= uint_metadata.width - 1u || j >= uint_metadata.height - 1u) {
        return;
    }

    let coeff = float_metadata.diffusion_rate * float_metadata.delta_time * float_metadata.cell_size * float_metadata.cell_size;
    density_new[idx_center(i, j)] = (density[idx_center(i, j)] +
        coeff * (density_new[idx_center(i + 1u, j)] +
                 density_new[idx_center(i - 1u, j)] +
                 density_new[idx_center(i, j + 1u)] +
                 density_new[idx_center(i, j - 1u)])
        ) / (1.0 + 4.0 * coeff);
}


@compute @workgroup_size(16, 16)
fn density_advect_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    let i = global_id.x;
    let j = global_id.y;

    // don't process edge cells
    if (i == 0u || j == 0u || i >= uint_metadata.width - 1u || j >= uint_metadata.height - 1u) {
        return;
    }

    // check this here?
    let dt0 = float_metadata.delta_time / float_metadata.cell_size;

    let idx = idx_center(i, j);

    var x = f32(i) - dt0 * u[idx];
    var y = f32(j) - dt0 * v[idx];

    // don't sample border cells
    x = clamp(x, 0.5, f32(uint_metadata.width) - 1.5);
    y = clamp(y, 0.5, f32(uint_metadata.height) - 1.5);

    density_new[idx] = sample_density(x, y);
}



@compute @workgroup_size(16, 16)
fn velocity_add_sources_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= uint_metadata.width || j >= uint_metadata.height) {
        return;
    }

    let idx = idx_center(i, j);

    // add velocity sources
    if (u_constants[idx] != 0.0) {
        u[idx] = u_constants[idx];
    } else {
        u[idx] += float_metadata.delta_time * u_sources[idx];
    }

    if (v_constants[idx] != 0.0) {
        v[idx] = v_constants[idx];
    } else {
        v[idx] += float_metadata.delta_time * v_sources[idx];
    }
}


@compute @workgroup_size(16, 16)
fn velocity_diffuse_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    let i = global_id.x;
    let j = global_id.y;

    // don't process edge cells
    if (i == 0u || j == 0u || i >= uint_metadata.width - 1u || j >= uint_metadata.height - 1u) {
        return;
    }

    let coeff = float_metadata.viscosity * float_metadata.delta_time * float_metadata.cell_size * float_metadata.cell_size;

    // diffuse u
    u_new[idx_center(i, j)] = (u[idx_center(i, j)] +
        coeff * (u_new[idx_center(i + 1u, j)] +
                 u_new[idx_center(i - 1u, j)] +
                 u_new[idx_center(i, j + 1u)] +
                 u_new[idx_center(i, j - 1u)])
        ) / (1.0 + 4.0 * coeff);

    // diffuse v
    v_new[idx_center(i, j)] = (v[idx_center(i, j)] +
        coeff * (v_new[idx_center(i + 1u, j)] +
                 v_new[idx_center(i - 1u, j)] +
                 v_new[idx_center(i, j + 1u)] +
                 v_new[idx_center(i, j - 1u)])
        ) / (1.0 + 4.0 * coeff);
}

@compute @workgroup_size(16, 16)
fn velocity_divergence_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    let i = global_id.x;
    let j = global_id.y;

    // don't process edge cells
    if (i == 0u || j == 0u || i >= uint_metadata.width - 1u || j >= uint_metadata.height - 1u) {
        return;
    }

    let idx = idx_center(i, j);

    let div = -0.5 * float_metadata.cell_size * (u[idx_center(i + 1u, j)] - u[idx_center(i - 1u, j)] +
               v[idx_center(i, j + 1u)] - v[idx_center(i, j - 1u)]);

    divergence[idx] = div;
    pressure[idx] = 0.0;
}


@compute @workgroup_size(16, 16)
fn velocity_pressure_solve_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    let i = global_id.x;
    let j = global_id.y;

    // don't process edge cells
    if (i == 0u || j == 0u || i >= uint_metadata.width - 1u || j >= uint_metadata.height - 1u) {
        return;
    }

    let idx = idx_center(i, j);
    pressure[idx] = (divergence[idx] +
        pressure[idx_center(i + 1u, j)] +
        pressure[idx_center(i - 1u, j)] +
        pressure[idx_center(i, j + 1u)] +
        pressure[idx_center(i, j - 1u)])
        / 4.0;

}


@compute @workgroup_size(16, 16)
fn velocity_project_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    let i = global_id.x;
    let j = global_id.y;

    // don't process edge cells
    if (i == 0u || j == 0u || i >= uint_metadata.width - 1u || j >= uint_metadata.height - 1u) {
        return;
    }

    let idx = idx_center(i, j);
    u[idx] -= 0.5 * (pressure[idx_center(i + 1u, j)] - pressure[idx_center(i - 1u, j)]) / float_metadata.cell_size;
    v[idx] -= 0.5 * (pressure[idx_center(i, j + 1u)] - pressure[idx_center(i, j - 1u)]) / float_metadata.cell_size;
}


@compute @workgroup_size(16, 16)
fn velocity_advect_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    let i = global_id.x;
    let j = global_id.y;

    // don't process edge cells
    if (i == 0u || j == 0u || i >= uint_metadata.width - 1u || j >= uint_metadata.height - 1u) {
        return;
    }

    // check this here?
    let dt0 = float_metadata.delta_time / float_metadata.cell_size;

    let idx = idx_center(i, j);

    var x = f32(i) - dt0 * u[idx];
    var y = f32(j) - dt0 * v[idx];

    // don't sample border cells
    x = clamp(x, 0.5, f32(uint_metadata.width) - 1.5);
    y = clamp(y, 0.5, f32(uint_metadata.height) - 1.5);

    u_new[idx] = sample_u(x, y);
    v_new[idx] = sample_v(x, y);
}


@compute @workgroup_size(16, 16)
fn set_boundary_scalar_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    let i = global_id.x;
    let j = global_id.y;

    // only process edge cells
    if (i > 0u && j > 0u && i < uint_metadata.width - 1u && j < uint_metadata.height - 1u) {
        return;
    }

    let idx = idx_center(i, j);
    let idx_interior = idx_center(
        clamp(i, 1u, uint_metadata.width - 2u),
        clamp(j, 1u, uint_metadata.height - 2u)
    );

    let btype = obstacles[idx];

    if (btype == 0u) { // outflow
        // set density and divergence to zero
        density[idx] = 0.0;
        density_new[idx] = 0.0;
        divergence[idx] = 0.0;

        // set pressure to zero gradient
        pressure[idx] = pressure[idx_interior];
        return;

    } else { // solid wall or inflow
        // copy from nearest interior cell
        density[idx] = density[idx_interior];
        density_new[idx] = density_new[idx_interior];
        pressure[idx] = pressure[idx_interior];
        divergence[idx] = divergence[idx_interior];
    }
}

@compute @workgroup_size(16, 16)
fn set_boundary_vector_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    let i = global_id.x;
    let j = global_id.y;

    // only process edge cells
    if (i > 0u && j > 0u && i < uint_metadata.width - 1u && j < uint_metadata.height - 1u) {
        return;
    }

    let idx = idx_center(i, j);
    let idx_interior = idx_center(
        clamp(i, 1u, uint_metadata.width - 2u),
        clamp(j, 1u, uint_metadata.height - 2u)
    );

    // velocity components (u, v)
    let btype = obstacles[idx];

    if (btype == 0u) { // outflow
        // copy from interior, no reflection
        u[idx] = u[idx_interior];
        v[idx] = v[idx_interior];
        u_new[idx] = u_new[idx_interior];
        v_new[idx] = v_new[idx_interior];

    } else { // solid wall
        // invert velocity component if on boundary
        if (i == 0u || i == uint_metadata.width - 1u) {
            u[idx] = -u[idx_interior];
            u_new[idx] = -u_new[idx_interior];
        } else {
            u[idx] = u[idx_interior];
            u_new[idx] = u_new[idx_interior];
        }

        if (j == 0u || j == uint_metadata.height - 1u) {
            v[idx] = -v[idx_interior];
            v_new[idx] = -v_new[idx_interior];
        } else {
            v[idx] = v[idx_interior];
            v_new[idx] = v_new[idx_interior];
        }
    }
}


// swap helpers
@compute @workgroup_size(16, 16)
fn swap_velocity_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= uint_metadata.width || j >= uint_metadata.height) {
        return;
    }

    let idx = idx_center(i, j);
    let temp_u = u[idx];
    u[idx] = u_new[idx];
    u_new[idx] = temp_u;

    let temp_v = v[idx];
    v[idx] = v_new[idx];
    v_new[idx] = temp_v;
}

@compute @workgroup_size(16, 16)
fn swap_density_main(
    @builtin(global_invocation_id) global_id : vec3u,
    @builtin(workgroup_id) workgroup_id : vec3u,
    @builtin(local_invocation_id) local_id : vec3u,
) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= uint_metadata.width || j >= uint_metadata.height) {
        return;
    }

    let idx = idx_center(i, j);
    let temp_density = density[idx];
    density[idx] = density_new[idx];
    density_new[idx] = temp_density;
}