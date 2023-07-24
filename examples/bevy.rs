use bevy::prelude::*;
use bevy_aabb_instancing::{
    Cuboid, CuboidMaterial, CuboidMaterialMap, Cuboids, ScalarHueOptions,
    VertexPullingRenderPlugin, COLOR_MODE_SCALAR_HUE,
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
        .add_plugin(VertexPullingRenderPlugin { outlines: true })
        .add_plugin(LookTransformPlugin)
        .add_plugin(UnrealCameraPlugin::default())
        .add_startup_system(setup)
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

    let grid = vdb_reader.read_grid::<half::f16>(&grid_to_load).unwrap();
    let instances: Vec<Cuboid> = grid
        .iter()
        .map(|(pos, voxel)| {
            Cuboid::new(
                pos * 0.1,
                (pos + Vec3::new(1.0, 1.0, 1.0)) * 0.1,
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
