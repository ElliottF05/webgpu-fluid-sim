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


// CONSTANTS

const RADIUS: f32 = 0.5;
const RESTITUTION: f32 = 0.8; // coefficient of restitution for collision response


// BINDINGS AND BUFFERS

@group(0) @binding(0) var<uniform> metadata: Metadata;

@group(0) @binding(1) var<storage, read_write> mass_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> pos_buf: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> vel_buf: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> pos_scratch: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> vel_scratch: array<vec2<f32>>;

@group(0) @binding(6) var<storage, read_write> body_indices: array<u32>;
@group(0) @binding(7) var<storage, read_write> node_data: array<NodeData>;


// HELPER FUNCTIONS

fn is_leaf(node_index: u32, n: u32) -> bool {
    return node_index >= n - 1u;
}

fn aabb_intersects_circle(aabb_min: vec2<f32>, aabb_max: vec2<f32>, center: vec2<f32>, radius: f32) -> bool {
    // Find the closest point on the AABB to the circle center
    let closest = clamp(center, aabb_min, aabb_max);
    let diff = center - closest;
    let dist_sq = dot(diff, diff);
    return dist_sq <= radius * radius;
}

fn resolve_collision(
    pos1: vec2<f32>, vel1: vec2<f32>, m1: f32,
    pos2: vec2<f32>, vel2: vec2<f32>, m2: f32
) -> vec2<f32> {
    // Compute impulse-based collision response
    // Returns the velocity change for body 1

    let n = normalize(pos1 - pos2); // collision normal pointing from 2 to 1
    let v_rel = vel1 - vel2; // relative velocity
    let v_rel_n = dot(v_rel, n); // relative velocity along normal

    // Only resolve if bodies are approaching
    if v_rel_n >= 0.0 {
        return vec2<f32>(0.0);
    }

    // Impulse magnitude using coefficient of restitution
    let j = -(1.0 + RESTITUTION) * v_rel_n / (1.0 / m1 + 1.0 / m2);

    // Velocity change for body 1
    return (j / m1) * n;
}


// MAIN COMPUTE SHADER - Detect collisions and write to scratch buffers

@compute @workgroup_size(64)
fn collision_detect_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let n = metadata.num_bodies;

    if thread_idx >= n {
        return;
    }

    let body_idx = body_indices[thread_idx];

    let pos1 = pos_buf[body_idx];
    let vel1 = vel_buf[body_idx];
    let m1 = mass_buf[body_idx];

    var vel_change = vec2<f32>(0.0);
    var pos_correction = vec2<f32>(0.0);
    var collision_count = 0u;

    // Traverse LBVH to find potential collisions
    const stack_size = 256;
    var stack: array<u32, stack_size>;
    var stack_ptr: i32 = 0;
    stack[stack_ptr] = 0u; // start with root node

    let collision_dist = 2.0 * RADIUS;
    let collision_dist_sq = collision_dist * collision_dist;

    while stack_ptr >= 0 {
        let node_idx = stack[stack_ptr];
        let node = node_data[node_idx];
        stack_ptr -= 1;

        if is_leaf(node_idx, n) {
            // Check collision with this leaf body
            let other_body_idx = body_indices[node_idx - (n - 1u)];

            if other_body_idx != body_idx {
                let pos2 = pos_buf[other_body_idx];
                let diff = pos1 - pos2;
                let dist_sq = dot(diff, diff);

                if dist_sq < collision_dist_sq && dist_sq > 0.0001 {
                    // Collision detected
                    let vel2 = vel_buf[other_body_idx];
                    let m2 = mass_buf[other_body_idx];

                    // Accumulate velocity change from impulse
                    vel_change += resolve_collision(pos1, vel1, m1, pos2, vel2, m2);

                    // Position correction to separate overlapping bodies
                    let dist = sqrt(dist_sq);
                    let overlap = collision_dist - dist;
                    if overlap > 0.0 {
                        let correction_dir = diff / dist;
                        // Each body gets half the correction, weighted by inverse mass
                        let total_inv_mass = 1.0 / m1 + 1.0 / m2;
                        pos_correction += correction_dir * overlap * (1.0 / m1) / total_inv_mass;
                    }

                    collision_count += 1u;
                }
            }
        } else {
            // Internal node - check if AABB (expanded by radius) intersects with our position
            let expanded_min = node.aabb_min - vec2<f32>(RADIUS);
            let expanded_max = node.aabb_max + vec2<f32>(RADIUS);

            if aabb_intersects_circle(expanded_min, expanded_max, pos1, RADIUS) {
                // Need to explore children
                if stack_ptr + 2 < stack_size {
                    stack_ptr += 1;
                    stack[stack_ptr] = node.left_child;
                    stack_ptr += 1;
                    stack[stack_ptr] = node.right_child;
                }
            }
        }
    }

    // Average the corrections if there were multiple collisions
    if collision_count > 0u {
        let count_f = f32(collision_count);
        vel_change = vel_change / count_f;
        pos_correction = pos_correction / count_f;
    }

    // Write results to scratch buffers
    vel_scratch[body_idx] = vel1 + vel_change;
    pos_scratch[body_idx] = pos1 + pos_correction;
}


// Copy scratch buffers back to main buffers

@compute @workgroup_size(64)
fn collision_apply_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let n = metadata.num_bodies;

    if thread_idx >= n {
        return;
    }

    let body_idx = body_indices[thread_idx];

    pos_buf[body_idx] = pos_scratch[body_idx];
    vel_buf[body_idx] = vel_scratch[body_idx];
}
