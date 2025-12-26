// STRUCTS

struct Metadata {
    num_bodies: u32,
    grav_constant: f32,
    delta_time: f32,
    epsilon_multiplier: f32,
    bh_theta: f32,
    _pad0: u32,
}

struct NodeData {
    center_of_mass: vec2<f32>,
    aabb_min: vec2<f32>,
    aabb_max: vec2<f32>,
    total_mass: f32,
    length: f32,
    left_child: u32,
    right_child: u32,
    parent: u32,
    _pad0: u32,
}


// BINDINGS AND BUFFERS

// metadata buffers
@group(0) @binding(0) var<uniform> metadata: Metadata;

// data buffers
@group(0) @binding(1) var<storage, read_write> mass_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pos_buf: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> vel_buf: array<vec2<f32>>;

// @group(0) @binding(4) var<storage, read_write> morton_codes: array<u32>; not needed here
@group(0) @binding(5) var<storage, read_write> body_indices: array<u32>;

@group(0) @binding(6) var<storage, read_write> node_data: array<NodeData>;
// @group(0) @binding(7) var<storage, read_write> node_status: array<atomic<u32>>; not needed here


// HELPER FUNCTIONS

fn is_leaf(node_index: u32, n: u32) -> bool {
    return node_index >= n - 1u;
}

@compute @workgroup_size(64)
fn bh_vel_step_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let n = metadata.num_bodies;
    if thread_idx >= n {
        return;
    }

    let body_idx = body_indices[thread_idx];

    let g = metadata.grav_constant;
    let dt = metadata.delta_time;
    let theta = metadata.bh_theta;

    let pos1 = pos_buf[body_idx];
    let m1 = mass_buf[body_idx];

    var accel = vec2<f32>(0.0, 0.0);

    const stack_size = 256;
    var stack: array<u32, stack_size>;
    var stack_ptr: i32 = 0; // always point to top element of stack
    stack[stack_ptr] = 0u; // start with root node

    while stack_ptr >= 0 {

        let node_idx = stack[stack_ptr];
        let node = node_data[node_idx];
        stack_ptr -= 1;

        let pos2 = node.center_of_mass;
        let m2 = node.total_mass;
        
        let r = pos2 - pos1;
        let dist_squared = dot(r, r);
        if is_leaf(node_idx, n) {
            let leaf_body_idx = body_indices[node_idx - (n - 1u)];
            if leaf_body_idx == body_idx {
                continue;
            }
        }

        let m_eff = max(m1, m2);
        let eps = metadata.epsilon_multiplier * sqrt(g * m_eff * dt);
        let inv_denom = inverseSqrt(dist_squared + eps * eps);

        if is_leaf(node_idx, n) || node.length * inv_denom < theta {
            // treat as single body
            let inv_denom_3 = inv_denom * inv_denom * inv_denom;
            accel += g * m2 * r * inv_denom_3;
        } else {
            // open the node
            if stack_ptr + 2 < stack_size {
                stack_ptr += 1;
                stack[stack_ptr] = node.left_child;
                stack_ptr += 1;
                stack[stack_ptr] = node.right_child;
            }
        }
    }
    
    // update velocity
    vel_buf[body_idx] = vel_buf[body_idx] + accel * dt;
}

@compute @workgroup_size(64)
fn bh_pos_step_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    if thread_idx >= metadata.num_bodies {
        return;
    }
    let body_idx = body_indices[thread_idx];
    pos_buf[body_idx] = pos_buf[body_idx] + vel_buf[body_idx] * metadata.delta_time;
}