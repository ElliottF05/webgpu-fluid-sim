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

struct NodeData {
    center_of_mass: vec2<f32>,
    aabb_min: vec2<f32>,
    aabb_max: vec2<f32>,
    total_mass: f32,
    length: f32,
    left_child: u32,
    right_child: u32,
    parent: u32,
    _pad: u32,
}


// BINDINGS AND BUFFERS

// metadata buffers
@group(0) @binding(0) var<uniform> float_metadata: FloatMetadata;
@group(0) @binding(1) var<uniform> uint_metadata: UintMetadata;

// data buffers
@group(0) @binding(2) var<storage, read_write> pos_buf: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> mass_buf: array<f32>;
@group(0) @binding(4) var<storage, read_write> morton_codes: array<u32>;
@group(0) @binding(5) var<storage, read_write> body_indices: array<u32>;

@group(0) @binding(6) var<storage, read_write> node_data: array<NodeData>;
@group(0) @binding(7) var<storage, read_write> node_status: array<atomic<u32>>;


@compute @workgroup_size(64)
fn compute_morton_codes_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= uint_metadata.num_bodies {
        return;
    }

    let scale = 2000.0; // side length of the square region we map to
    let half = 0.5 * scale;

    let pos = pos_buf[i];
    var uv = (pos + vec2<f32>(half)) / vec2<f32>(scale); // map from [-half, half] to [0,1]
    uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)); // clamp to [0, 1]

    let x = min(u32(uv.x * 65535.0 + 0.5), 65535u); // 16 bits
    let y = min(u32(uv.y * 65535.0 + 0.5), 65535u); // 16 bits

    var morton_code = 0u;
    for (var bit = 0u; bit < 16u; bit += 1u) {
        let mask = 1u << bit;
        let bit_x = ((x & mask) >> bit) << (2u * bit);
        let bit_y = ((y & mask) >> bit) << (2u * bit + 1u);
        morton_code = morton_code | bit_x | bit_y;
    }

    morton_codes[i] = morton_code;
    body_indices[i] = i;
}


fn delta(i: i32, j: i32) -> i32 {
    if j < 0 || j >= i32(uint_metadata.num_bodies) {
        return -1;
    }
    let a = morton_codes[u32(i)];
    let b = morton_codes[u32(j)];
    if a == b {
        // tie break using indices
        return 32 + i32(countLeadingZeros(u32(i) ^ u32(j)));
    }
    return i32(countLeadingZeros(a ^ b));
}

@compute @workgroup_size(64)
fn build_lbvh_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // this exactly follows figure 4 in
    // https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2012/11/karras2012hpg_paper.pdf

    let i_u = global_id.x;
    let i = i32(i_u);
    let n = uint_metadata.num_bodies;

    // internal nodes are indexed [0, n-2], leaves are [n-1, 2n-2]
    if i_u >= n - 1u {
        return;
    }

    // determine direction of the range
    let delta_left = delta(i, i - 1);
    let delta_right = delta(i, i + 1);
    let d = select(-1, 1, delta(i, i + 1) > delta(i, i - 1));

    // compute upper bound for the length of the range
    let delta_min = delta(i, i - d);
    var l_max = 2;
    while delta(i, i + d * l_max) > delta_min {
        l_max = l_max * 2;
    }

    // find the other end using binary search
    var l = 0;
    var t = l_max / 2;
    while t >= 1 {
        if delta(i, i + d * (l + t)) > delta_min {
            l = l + t;
        }
        t = t / 2;
    }
    let j = i + d * l;
    let j_u = u32(j);

    // find the split position using binary search
    let delta_node = delta(i, j);
    var s = 0;
    t = (l + 1) / 2; // ceil division
    loop {
        if delta(i, i + d * (s + t)) > delta_node {
            s = s + t;
        }
        if t == 1 {
            break;
        }
        t = (t + 1) / 2;
    }
    let gamma = i + d * s + min(d, 0);
    let gamma_u = u32(gamma);


    // get left child
    if min(i, j) == gamma { // child is leaf node
        node_data[i_u].left_child = (n - 1u) + gamma_u;
    } else {
        node_data[i_u].left_child = gamma_u;
    }

    // get right child
    if max(i, j) == gamma + 1 { // child is leaf node
        node_data[i_u].right_child = (n - 1u) + (gamma_u + 1u);
    } else {  
        node_data[i_u].right_child = gamma_u + 1u;
    }

    // set parent pointers
    let left_idx = node_data[i_u].left_child;
    let right_idx = node_data[i_u].right_child;
    node_data[left_idx].parent = i_u;
    node_data[right_idx].parent = i_u;
    if i_u == 0u {
        node_data[i_u].parent = 0xFFFFFFFFu; // root has no parent
    }

    // set to unvisited
    atomicStore(&node_status[i_u], 0u);
}


@compute @workgroup_size(64)
fn fill_lbvh_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = uint_metadata.num_bodies;
    var curr_idx = (n - 1u) + global_id.x; // leaves are indexed [n-1, 2n-2]

    if curr_idx < (n - 1u) || curr_idx >= (2u * n - 1u) {
        return;
    }

    // initialize leaf node data
    let body_idx = body_indices[curr_idx - (n - 1u)];
    let mass = mass_buf[body_idx];
    let pos = pos_buf[body_idx];

    node_data[curr_idx].total_mass = mass;
    node_data[curr_idx].center_of_mass = pos;
    node_data[curr_idx].aabb_min = pos;
    node_data[curr_idx].aabb_max = pos;
    node_data[curr_idx].length = 0.0;


    // propagate up the tree
    var count = 0u;
    while curr_idx != 0u {
        count = count + 1u;
        if count > 200000u {
            break; // prevent infinite loops
        }
        let parent_idx = node_data[curr_idx].parent;

        if parent_idx == curr_idx {
            break; // reached root
        }
        if parent_idx >= (n - 1u) {
            break; // should not happen, but just in case
        }

        // attempt to acquire lock on parent
        let prev_status = atomicAdd(&node_status[parent_idx], 1u);
        if prev_status == 0u {
            // we are the first to arrive at this parent so break
            break;
        }

        // we are the second to arrive at this parent, so we can compute its data
        let left_idx = node_data[parent_idx].left_child;
        let right_idx = node_data[parent_idx].right_child;

        let left_data = node_data[left_idx];
        let right_data = node_data[right_idx];

        // combine AABBs
        node_data[parent_idx].aabb_min = min(left_data.aabb_min, right_data.aabb_min);
        node_data[parent_idx].aabb_max = max(left_data.aabb_max, right_data.aabb_max);

        // combine masses and centers of mass
        let total_mass = left_data.total_mass + right_data.total_mass;
        node_data[parent_idx].total_mass = total_mass;
        if total_mass > 0.0 {
            node_data[parent_idx].center_of_mass =
                (left_data.center_of_mass * left_data.total_mass +
                 right_data.center_of_mass * right_data.total_mass) / total_mass;
        } else {
            node_data[parent_idx].center_of_mass = vec2<f32>(0.0);
        }

        // compute length of longest side of AABB
        let aabb_size = node_data[parent_idx].aabb_max - node_data[parent_idx].aabb_min;
        node_data[parent_idx].length = max(aabb_size.x, aabb_size.y);

        // move up the tree
        curr_idx = parent_idx;
    }
}