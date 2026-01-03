struct Params {
    dst_w: u32,
    dst_h: u32,
    src_w: u32,
    src_h: u32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if (x >= params.dst_w || y >= params.dst_h) {
        return;
    }
    let u = (f32(x) + 0.5) / f32(params.dst_w);
    let v = (f32(y) + 0.5) / f32(params.dst_h);
    let color = textureSampleLevel(src_tex, samp, vec2<f32>(u, v), 0.0);
    textureStore(dst_tex, vec2<i32>(i32(x), i32(y)), color);
}
