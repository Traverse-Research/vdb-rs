use bevy::prelude::*;
use bevy_aabb_instancing::{
    Cuboid, CuboidMaterial, CuboidMaterialMap, Cuboids, ScalarHueOptions,
    VertexPullingRenderPlugin, COLOR_MODE_SCALAR_HUE,
};
use half::f16;
use rav1e::{
    config::SpeedSettings, prelude::v_frame, Config, Context, EncoderConfig, EncoderStatus,
};
use rav1d_safe::src::managed::{Decoder, Planes};
use smooth_bevy_cameras::{controllers::unreal::*, LookTransformPlugin};
use vdb_rs::VdbReader;

use std::{error::Error, fs::File, io::BufReader};

fn main() -> Result<(), Box<dyn Error>> {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "VDB Viewer".into(),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(VertexPullingRenderPlugin { outlines: true })
        .add_plugins(LookTransformPlugin)
        .add_plugins(UnrealCameraPlugin::default())
        .add_systems(Startup, setup)
        .run();

    Ok(())
}

/// set up a simple 3D scene
fn setup(mut commands: Commands, mut color_options_map: ResMut<CuboidMaterialMap>) {
    let color_options_id = color_options_map.push(CuboidMaterial {
        color_mode: COLOR_MODE_SCALAR_HUE,
        scalar_hue: ScalarHueOptions {
            min_visible: -10000.0,
            max_visible: 10000.0,
            clamp_min: -1.0,
            clamp_max: 0.5,
            ..Default::default()
        },
        ..Default::default()
    });

    let filename = std::env::args()
        .nth(1)
        .expect("Missing VDB filename as first argument");

    let f = File::open(filename).unwrap();
    let mut vdb_reader = VdbReader::new(BufReader::new(f)).unwrap();
    let grid_names = vdb_reader.available_grids();

    let _grid_to_load = std::env::args().nth(2).unwrap_or_else(|| {
        println!(
            "Grid name not specified, defaulting to first available grid.\nAvailable grids: {:?}",
            grid_names
        );
        grid_names.first().cloned().unwrap_or(String::new())
    });

    let max_aabb = vdb_reader.grid_descriptors[grid_names.first().unwrap()]
        .aabb_max()
        .unwrap();
    let min_aabb = vdb_reader.grid_descriptors[grid_names.first().unwrap()]
        .aabb_min()
        .unwrap();
    let extent = max_aabb - min_aabb + 1;

    info!("min_aabb: {}, max_aabb: {}, extent: {}", min_aabb, max_aabb, extent);

    // ENCODE: VDB -> AV1/IVF
    {
        let grid = vdb_reader
            .read_grid::<half::f16>(&grid_names.first().unwrap())
            .unwrap();

        let mut voxel_grid = vec![f16::ZERO; (extent.x * extent.y * extent.z) as usize];
        grid.iter().for_each(|(pos, voxel, level)| {
            for z in 0..level.scale() as u32 {
                for y in 0..level.scale() as u32 {
                    for x in 0..level.scale() as u32 {
                        if pos.x < 0.0 || pos.y < 0.0 || pos.z < 0.0 {
                            dbg!("Error pos: {}", pos);
                        }
                        let actual_index =
                            (pos + Vec3::new(x as f32, y as f32, z as f32)).as_uvec3()
                                - min_aabb.as_uvec3();
                        let id = actual_index.x
                            + actual_index.y * extent.x as u32
                            + actual_index.z * extent.x as u32 * extent.y as u32;
                        if id >= voxel_grid.len() as u32 {
                            dbg!(pos, x, y, z);
                            dbg!(extent);
                        }
                        voxel_grid[id as usize] = voxel;
                    }
                }
            }
        });

        let width = extent.x as usize;
        let height = extent.y as usize;
        let slices = extent.z as usize;

        let enc = EncoderConfig {
            width,
            height,
            speed_settings: SpeedSettings::from_preset(9),
            bit_depth: 8,
            chroma_sampling: rav1e::color::ChromaSampling::Cs400,
            ..Default::default()
        };

        let cfg = Config::new().with_encoder_config(enc);

        let mut ctx: Context<u16> = cfg.new_context().unwrap();

        let f = ctx.new_frame();

        let mut out = std::fs::File::create("out.ivf").unwrap();

        ivf::write_ivf_header(&mut out, width, height, 30, 1);
        let mut frame_idx = 0;
        loop {
            match ctx.receive_packet() {
                Ok(pkt) => {
                    info!("Packet {}", pkt.input_frameno);
                    ivf::write_ivf_frame(&mut out, pkt.input_frameno, &pkt.data);
                }
                Err(EncoderStatus::Encoded) => (),
                Err(EncoderStatus::LimitReached) => {
                    break;
                }
                Err(EncoderStatus::NeedMoreData) => {
                    if frame_idx >= slices {
                        ctx.flush();
                        continue;
                    }
                    info!("Writing frame {} out of {}", frame_idx, slices);

                    let mut frame = f.clone();
                    for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
                        if plane_idx != 0 {
                            continue;
                        }
                        for (row_idx, dst_row) in plane
                            .mut_slice(v_frame::plane::PlaneOffset::default())
                            .rows_iter_mut()
                            .enumerate()
                            .take(height)
                        {
                            for (col, dst) in dst_row.iter_mut().take(width).enumerate() {
                                let multiplier = ((1 << 8) - 1) as f32;
                                let id = col + row_idx * width + frame_idx * width * height;
                                *dst = (voxel_grid[id].to_f32() * multiplier * 10.0)
                                    .clamp(0.0, 255.0)
                                    as u16;
                            }
                        }
                    }

                    match ctx.send_frame(frame) {
                        Ok(_) => {}
                        Err(EncoderStatus::EnoughData) => {
                            info!("Unable to append frame {} to the internal queue", frame_idx)
                        }
                        Err(e) => {
                            panic!("Unable to send frame {frame_idx}: {e:?}");
                        }
                    }
                    frame_idx += 1;
                }
                _ => {}
            }
        }
    }

    // DECODE: IVF/AV1 -> voxel grid
    let decoded_voxels = {
        let mut ivf_file =
            BufReader::new(std::fs::File::open("out.ivf").expect("Failed to open out.ivf"));

        let header = ivf::read_header(&mut ivf_file).expect("Failed to read IVF header");
        let width = header.w as usize;
        let height = header.h as usize;
        let slices = extent.z as usize;

        info!(
            "Decoding IVF: {}x{}, expecting {} slices",
            width, height, slices
        );

        let mut decoder = Decoder::new().expect("Failed to create rav1d decoder");

        let mut decoded_grid = vec![f16::ZERO; width * height * slices];
        let mut frames_decoded = 0usize;

        // Read all IVF packets
        let mut packets = Vec::new();
        loop {
            match ivf::read_packet(&mut ivf_file) {
                Ok(pkt) => packets.push(pkt),
                Err(_) => break,
            }
        }
        info!("Read {} packets from IVF", packets.len());

        for pkt in &packets {
            match decoder.decode(&pkt.data) {
                Ok(Some(frame)) => {
                    decode_frame_to_grid(&frame, &mut decoded_grid, width, height, slices, frames_decoded);
                    frames_decoded += 1;
                    info!("Decoded frame {} / {}", frames_decoded, slices);
                }
                Ok(None) => {}
                Err(e) => panic!("Decode error: {e}"),
            }

            // Drain any buffered frames
            loop {
                match decoder.get_frame() {
                    Ok(Some(frame)) => {
                        decode_frame_to_grid(&frame, &mut decoded_grid, width, height, slices, frames_decoded);
                        frames_decoded += 1;
                        info!("Decoded frame {} / {}", frames_decoded, slices);
                    }
                    Ok(None) => break,
                    Err(e) => panic!("Decode error draining: {e}"),
                }
            }
        }

        // Flush remaining frames
        match decoder.flush() {
            Ok(remaining_frames) => {
                for frame in &remaining_frames {
                    decode_frame_to_grid(frame, &mut decoded_grid, width, height, slices, frames_decoded);
                    frames_decoded += 1;
                    info!("Decoded frame {} / {} (flushed)", frames_decoded, slices);
                }
            }
            Err(e) => info!("Flush completed with: {e}"),
        }

        info!(
            "Decoding complete: {} frames decoded, grid size: {}",
            frames_decoded,
            decoded_grid.len()
        );

        decoded_grid
    };

    // Visualize the decoded voxels
    let width = extent.x as usize;
    let height = extent.y as usize;
    let slices = extent.z as usize;

    // Use a threshold to filter out AV1 lossy compression artifacts.
    // AV1 at speed 9 creates ringing at boundaries; pixel values of ~5-15
    // are common artifacts, which decode to 0.002-0.006. Use 0.005 to be safe.
    let artifact_threshold = 0.008;

    let mut instances: Vec<Cuboid> = Vec::new();
    for z in 0..slices {
        for y in 0..height {
            for x in 0..width {
                let id = x + y * width + z * width * height;
                let voxel = decoded_voxels[id];
                if voxel.to_f32() > artifact_threshold {
                    let pos = Vec3::new(
                        x as f32 + min_aabb.x as f32,
                        y as f32 + min_aabb.y as f32,
                        z as f32 + min_aabb.z as f32,
                    );
                    instances.push(Cuboid::new(
                        pos * 0.1,
                        (pos + 1.0) * 0.1,
                        u32::from_le_bytes(f32::to_le_bytes(voxel.to_f32())),
                    ));
                }
            }
        }
    }

    info!("Rendering {} decoded voxels", instances.len());

    let cuboids = Cuboids::new(instances);
    let aabb = cuboids.aabb();
    commands
        .spawn(SpatialBundle::default())
        .insert((cuboids, aabb, color_options_id));

    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    commands
        .spawn(Camera3dBundle::default())
        .insert(UnrealCameraBundle::new(
            UnrealCameraController::default(),
            Vec3::new(0.0, 1.0, 10.0),
            Vec3::ZERO,
            Vec3::Y,
        ));
}

fn decode_frame_to_grid(
    frame: &rav1d_safe::src::managed::Frame,
    decoded_grid: &mut [f16],
    width: usize,
    height: usize,
    slices: usize,
    frame_idx: usize,
) {
    if frame_idx >= slices {
        return;
    }

    let multiplier = ((1 << 8) - 1) as f32;

    match frame.planes() {
        Planes::Depth8(planes) => {
            let y_plane = planes.y();
            for row in 0..y_plane.height().min(height) {
                let row_data = y_plane.row(row);
                for col in 0..row_data.len().min(width) {
                    let pixel = row_data[col];
                    // Reverse the encoding: encoded = (f16 * 255.0 * 10.0).clamp(0, 255)
                    // So: f16 = pixel / 255.0 / 10.0
                    let value = pixel as f32 / multiplier / 10.0;
                    let id = col + row * width + frame_idx * width * height;
                    decoded_grid[id] = f16::from_f32(value);
                }
            }
        }
        Planes::Depth16(planes) => {
            let y_plane = planes.y();
            for row in 0..y_plane.height().min(height) {
                let row_data = y_plane.row(row);
                for col in 0..row_data.len().min(width) {
                    let pixel = row_data[col];
                    let value = pixel as f32 / multiplier / 10.0;
                    let id = col + row * width + frame_idx * width * height;
                    decoded_grid[id] = f16::from_f32(value);
                }
            }
        }
    }
}
