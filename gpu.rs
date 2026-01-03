use crate::error::{Error, Result};

#[cfg(target_os = "windows")]
mod d3d_gpu {
    use super::{Error, Result};
    use image::{RgbImage, RgbaImage};
    use std::sync::OnceLock;
    use wgpu::util::DeviceExt;

    struct GpuContext {
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();
    static PREPROCESS_ENABLED: OnceLock<bool> = OnceLock::new();

    pub(super) fn preprocess_enabled() -> bool {
        *PREPROCESS_ENABLED.get_or_init(|| {
            std::env::var("PHOTO_TAGGER_GPU_PREPROCESS")
                .ok()
                .map(|v| {
                    let v = v.to_ascii_lowercase();
                    v == "1" || v == "true" || v == "yes"
                })
                .unwrap_or(false)
        })
    }

    fn get_context() -> Option<&'static GpuContext> {
        if !preprocess_enabled() {
            return None;
        }
        GPU_CONTEXT
            .get_or_init(|| match init_gpu() {
                Ok(ctx) => Some(ctx),
                Err(err) => {
                    log::warn!("GPU init failed; using CPU: {err}");
                    None
                }
            })
            .as_ref()
    }

    fn init_gpu() -> Result<GpuContext> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            flags: wgpu::InstanceFlags::empty(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok_or_else(|| Error::Init("No GPU adapter available".into()))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("photo-tag-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .map_err(|e| Error::Init(format!("GPU device request failed: {e}")))?;
        Ok(GpuContext { device, queue })
    }

    pub fn histogram_embedding(resized: &RgbImage) -> Result<Vec<f32>> {
        let Some(ctx) = get_context() else {
            return Err(Error::Init("GPU context unavailable".into()));
        };
        let pixel_count = (resized.width() * resized.height()) as usize;
        if pixel_count == 0 {
            return Ok(vec![0.0; 48]);
        }
        let mut packed = Vec::with_capacity(pixel_count);
        for p in resized.pixels() {
            let r = p[0] as u32;
            let g = p[1] as u32;
            let b = p[2] as u32;
            let a = 255u32;
            packed.push(r | (g << 8) | (b << 16) | (a << 24));
        }

        let device = &ctx.device;
        let queue = &ctx.queue;

        let pixels_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("embedding-pixels"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let zeros = vec![0u32; 48];
        let hist_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("embedding-hist"),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let count_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("embedding-count"),
            contents: bytemuck::cast_slice(&[pixel_count as u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("embedding-histogram"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/histogram.wgsl").into()),
        });
        let bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("embedding-bind-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("embedding-pipeline-layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("embedding-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embedding-bind-group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pixels_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: count_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embedding-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (pixel_count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embedding-staging"),
            size: (48 * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&hist_buf, 0, &staging, 0, (48 * 4) as u64);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| Error::Init(format!("GPU readback failed: {e}")))?
            .map_err(|e| Error::Init(format!("GPU readback failed: {e}")))?;
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let bins: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(bins.into_iter().map(|v| v as f32).collect())
    }

    pub fn resize_rgba8(
        input: &RgbaImage,
        dst_w: u32,
        dst_h: u32,
    ) -> Result<RgbaImage> {
        let Some(ctx) = get_context() else {
            return Err(Error::Init("GPU context unavailable".into()));
        };
        if dst_w == 0 || dst_h == 0 {
            return Err(Error::Init("Invalid resize target".into()));
        }
        let device = &ctx.device;
        let queue = &ctx.queue;

        let src_size = wgpu::Extent3d {
            width: input.width(),
            height: input.height(),
            depth_or_array_layers: 1,
        };
        let dst_size = wgpu::Extent3d {
            width: dst_w,
            height: dst_h,
            depth_or_array_layers: 1,
        };
        let src_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("resize-src"),
            size: src_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &src_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            input,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * input.width()),
                rows_per_image: Some(input.height()),
            },
            src_size,
        );
        let dst_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("resize-dst"),
            size: dst_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let src_view = src_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let dst_view = dst_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("resize-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resize-params"),
            contents: bytemuck::cast_slice(&[dst_w, dst_h, input.width(), input.height()]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resize-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/resize.wgsl").into()),
        });
        let bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("resize-bind-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resize-pipeline-layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("resize-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resize-bind-group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&dst_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resize-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let x_groups = (dst_w + 7) / 8;
            let y_groups = (dst_h + 7) / 8;
            pass.dispatch_workgroups(x_groups, y_groups, 1);
        }

        let padded_bytes_per_row = ((dst_w * 4 + 255) / 256) * 256;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resize-output"),
            size: (padded_bytes_per_row * dst_h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &dst_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(dst_h),
                },
            },
            dst_size,
        );
        queue.submit(Some(encoder.finish()));

        let slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| Error::Init(format!("GPU readback failed: {e}")))?
            .map_err(|e| Error::Init(format!("GPU readback failed: {e}")))?;
        device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let mut pixels = vec![0u8; (dst_w * dst_h * 4) as usize];
        for row in 0..dst_h as usize {
            let src_offset = row * padded_bytes_per_row as usize;
            let dst_offset = row * (dst_w * 4) as usize;
            let src_slice = &data[src_offset..src_offset + (dst_w * 4) as usize];
            pixels[dst_offset..dst_offset + (dst_w * 4) as usize].copy_from_slice(src_slice);
        }
        drop(data);
        output_buffer.unmap();

        let out = RgbaImage::from_vec(dst_w, dst_h, pixels)
            .ok_or_else(|| Error::Init("Failed to build resized image".into()))?;
        Ok(out)
    }
}

#[cfg(target_os = "windows")]
pub fn histogram_embedding(resized: &image::RgbImage) -> Result<Vec<f32>> {
    d3d_gpu::histogram_embedding(resized)
}

#[cfg(not(target_os = "windows"))]
pub fn histogram_embedding(_resized: &image::RgbImage) -> Result<Vec<f32>> {
    Err(Error::Init("GPU histogram unsupported on this OS".into()))
}

#[cfg(target_os = "windows")]
pub fn resize_rgba8(
    input: &image::RgbaImage,
    dst_w: u32,
    dst_h: u32,
) -> Result<image::RgbaImage> {
    d3d_gpu::resize_rgba8(input, dst_w, dst_h)
}

#[cfg(not(target_os = "windows"))]
pub fn resize_rgba8(
    _input: &image::RgbaImage,
    _dst_w: u32,
    _dst_h: u32,
) -> Result<image::RgbaImage> {
    Err(Error::Init("GPU resize unsupported on this OS".into()))
}

pub fn gpu_preprocess_enabled() -> bool {
    #[cfg(target_os = "windows")]
    {
        return d3d_gpu::preprocess_enabled();
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}
