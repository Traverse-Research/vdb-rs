use bevy::prelude::*;
use bevy_aabb_instancing::{
    Cuboid, CuboidMaterial, CuboidMaterialMap, Cuboids, ScalarHueOptions,
    VertexPullingRenderPlugin, COLOR_MODE_SCALAR_HUE,
};
use half::f16;
use rav1e::{
    config::SpeedSettings, prelude::v_frame, Config, Context, EncoderConfig, EncoderStatus,
};
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

    let grid_to_load = std::env::args().nth(2).unwrap_or_else(|| {
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

    info!("Extent: {}", extent);

    // MY GLORIOUS COMPRESSION
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
                            (pos + Vec3::new(x as f32, y as f32, z as f32)).as_uvec3();
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
                    ctx.flush();
                    break;
                }
                Err(EncoderStatus::NeedMoreData) => {
                    info!("Requesting more data");
                    frame_idx += 1;
                    if frame_idx >= slices {
                        ctx.flush();

                        // we need to continue the loop here to receive more packets
                        continue;
                    }
                    info!("Writing frame {} out of {}", frame_idx, slices);

                    let mut frame = f.clone();
                    for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
                        info!("Plane idx: {}", plane_idx);
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
                                if voxel_grid[id] != f16::ZERO {
                                    // dbg!(
                                    //     col,
                                    //     row_idx,
                                    //     frame_idx,
                                    //     id,
                                    //     voxel_grid[id],
                                    //     voxel_grid[id].to_f32() * multiplier
                                    // );
                                }
                                match plane_idx {
                                    0 => {
                                        *dst = (voxel_grid[id].to_f32() * multiplier * 10.0)
                                            .clamp(0.0, 255.0)
                                            as u16
                                    }
                                    1 => {
                                        *dst = (voxel_grid[id].to_f32() * multiplier * 10.0)
                                            .clamp(0.0, 255.0)
                                            as u16
                                    }
                                    2 => {
                                        *dst = (voxel_grid[id].to_f32() * multiplier * 10.0)
                                            .clamp(0.0, 255.0)
                                            as u16
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }

                    info!("Sending frame");
                    match ctx.send_frame(frame) {
                        Ok(_) => {}
                        Err(EncoderStatus::EnoughData) => {
                            info!("Unable to append frame {} to the internal queue", frame_idx)
                        }
                        Err(e) => {
                            panic!("Unable to send frame {frame_idx}: {e:?}");
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let grid = vdb_reader.read_grid::<half::f16>(&grid_to_load).unwrap();
    let instances: Vec<Cuboid> = grid
        .iter()
        .map(|(pos, voxel, level)| {
            Cuboid::new(
                pos * 0.1,
                (pos + level.scale()) * 0.1,
                u32::from_le_bytes(f32::to_le_bytes(voxel.to_f32())),
            )
        })
        .collect();
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
