struct Params {
    count: u32,
};

@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> hist: array<atomic<u32>, 48>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.count) {
        return;
    }
    let p = pixels[idx];
    let r = p & 255u;
    let g = (p >> 8u) & 255u;
    let b = (p >> 16u) & 255u;
    let bin_r = (r * 16u) / 256u;
    let bin_g = (g * 16u) / 256u;
    let bin_b = (b * 16u) / 256u;
    atomicAdd(&hist[bin_r], 1u);
    atomicAdd(&hist[16u + bin_g], 1u);
    atomicAdd(&hist[32u + bin_b], 1u);
}
