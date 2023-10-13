use bevy::{prelude::*, render::primitives::Aabb};
use bevy_aabb_instancing::{
    Cuboid, CuboidMaterial, CuboidMaterialId, CuboidMaterialMap, Cuboids, ScalarHueOptions,
    VertexPullingRenderPlugin, COLOR_MODE_SCALAR_HUE,
};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use glam::vec3;
use half::f16;
use smooth_bevy_cameras::{
    controllers::orbit::{OrbitCameraBundle, OrbitCameraController, OrbitCameraPlugin},
    LookTransformPlugin,
};
use vdb_rs::{Grid, Map, VdbLevel, VdbReader};

use std::{error::Error, fs::File, io::BufReader};

#[derive(Debug, PartialEq, Copy, Clone)]
enum SliceAxis {
    X = 0,
    Y,
    Z,
}

impl SliceAxis {
    pub fn unit_vec(self) -> Vec3 {
        match self {
            SliceAxis::X => vec3(1.0, 0.0, 0.0),
            SliceAxis::Y => vec3(0.0, 1.0, 0.0),
            SliceAxis::Z => vec3(0.0, 0.0, 1.0),
        }
    }
}

#[derive(Debug, PartialEq)]
enum RenderMode {
    FirstDensity,
    Tiles,
    Slice(SliceAxis),
}

#[derive(Resource)]
struct RenderSettings {
    render_mode: RenderMode,
    render_slice_index: i32,
    min_slice_indices: IVec3,
    max_slice_indices: IVec3,
    dirty: bool,
    visible_voxels: u64,
}

#[derive(Resource)]
struct ModelData {
    color_options_id: CuboidMaterialId,
    grid: Grid<f16>,
}

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
        .add_plugins(OrbitCameraPlugin::default())
        .add_systems(Startup, setup)
        .add_plugins(EguiPlugin)
        .add_systems(Update, settings_ui)
        .add_systems(Update, rebuild_model)
        .run();

    Ok(())
}

fn settings_ui(mut contexts: EguiContexts, mut settings: ResMut<RenderSettings>) {
    egui::Window::new("Settings").show(contexts.ctx_mut(), |ui| {
        egui::ComboBox::from_label("Render")
            .selected_text(format!("{:?}", settings.render_mode))
            .show_ui(ui, |ui| {
                settings.dirty |= ui
                    .selectable_value(
                        &mut settings.render_mode,
                        RenderMode::FirstDensity,
                        "Density",
                    )
                    .changed();
                settings.dirty |= ui
                    .selectable_value(&mut settings.render_mode, RenderMode::Tiles, "Tiles")
                    .changed();
                settings.dirty |= ui
                    .selectable_value(
                        &mut settings.render_mode,
                        RenderMode::Slice(SliceAxis::X),
                        "X slice",
                    )
                    .changed();
                settings.dirty |= ui
                    .selectable_value(
                        &mut settings.render_mode,
                        RenderMode::Slice(SliceAxis::Y),
                        "Y slice",
                    )
                    .changed();
                settings.dirty |= ui
                    .selectable_value(
                        &mut settings.render_mode,
                        RenderMode::Slice(SliceAxis::Z),
                        "Z slice",
                    )
                    .changed();
            });
        if let RenderMode::Slice(i) = settings.render_mode {
            let range =
                settings.min_slice_indices[i as usize]..=settings.max_slice_indices[i as usize];
            settings.dirty |= ui
                .add(egui::Slider::new(&mut settings.render_slice_index, range))
                .changed()
        }
        ui.label(format!("visible voxels: {}", settings.visible_voxels));
    });
}

fn rebuild_model(
    mut commands: Commands,
    mut settings: ResMut<RenderSettings>,
    model_data: Res<ModelData>,
    existing_voxels: Query<Entity, With<Aabb>>,
) {
    if settings.dirty {
        existing_voxels.for_each(|entity| {
            commands.entity(entity).despawn();
        });

        let translation = match model_data.grid.transform {
            Map::ScaleTranslateMap { translation, .. } => translation.as_vec3(),
            _ => vec3(0.0, 0.0, 0.0),
        };

        let slice_index = settings.render_slice_index;

        let reject_fn: Box<dyn Fn(Vec3, VdbLevel) -> bool> = match settings.render_mode {
            RenderMode::FirstDensity => Box::new(|_, _| false),
            RenderMode::Slice(i) => Box::new(move |pos, _| pos[i as usize] as i32 != slice_index),
            RenderMode::Tiles => Box::new(|_, level| level == VdbLevel::Voxel),
        };

        let instances: Vec<Cuboid> = model_data
            .grid
            .iter()
            .filter_map(|(mut pos, voxel, level)| {
                let level_invariate_position = (pos / level.scale()).floor() * level.scale()
                    + (slice_index % level.scale() as i32) as f32;
                if reject_fn(level_invariate_position, level) {
                    None
                } else {
                    let mut dimension_mult = Vec3::ONE;
                    if let RenderMode::Slice(i) = settings.render_mode {
                        dimension_mult -= i.unit_vec();
                        pos[i as usize] = slice_index as f32;
                    }
                    dimension_mult = (dimension_mult * level.scale()).max(Vec3::ONE);

                    let pos = pos + translation;
                    Some(Cuboid::new(
                        pos * 0.1,
                        (pos + Vec3::new(1.0, 1.0, 1.0) * dimension_mult) * 0.1,
                        u32::from_le_bytes(f32::to_le_bytes(voxel.to_f32())),
                    ))
                }
            })
            .collect();

        settings.visible_voxels = instances.len() as u64;
        let cuboids = Cuboids::new(instances);

        let aabb = cuboids.aabb();
        commands.spawn(SpatialBundle::default()).insert((
            cuboids,
            aabb,
            model_data.color_options_id,
        ));
        settings.dirty = false;
    }
}

/// set up a simple 3D scene
fn setup(mut commands: Commands, mut color_options_map: ResMut<CuboidMaterialMap>) {
    let grid = load_grid();

    commands.insert_resource(RenderSettings {
        render_mode: RenderMode::FirstDensity,
        render_slice_index: 0,
        min_slice_indices: grid.descriptor.aabb_min().unwrap(),
        max_slice_indices: grid.descriptor.aabb_max().unwrap(),
        dirty: true,
        visible_voxels: 0,
    });

    commands.insert_resource(ModelData {
        color_options_id: color_options_map.push(CuboidMaterial {
            color_mode: COLOR_MODE_SCALAR_HUE,
            scalar_hue: ScalarHueOptions {
                min_visible: -10000.0,
                max_visible: 10000.0,
                clamp_min: -1.0,
                clamp_max: 0.5,
                ..Default::default()
            },
            ..Default::default()
        }),
        grid,
    });
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
        .insert(OrbitCameraBundle::new(
            OrbitCameraController::default(),
            Vec3::new(0.0, 1.0, 10.0),
            Vec3::ZERO,
            Vec3::Y,
        ));
}

fn load_grid() -> Grid<f16> {
    let filename = std::env::args()
        .nth(1)
        .expect("Missing VDB filename as first argument");

    let f = File::open(filename.clone()).unwrap();
    let mut vdb_reader = VdbReader::new(BufReader::new(f)).unwrap();
    let grid_names = vdb_reader.available_grids();

    let grid_to_load = std::env::args().nth(2).unwrap_or_else(|| {
        println!(
            "Grid name not specified, defaulting to first available grid.\nAvailable grids: {:?}",
            grid_names
        );
        grid_names.first().cloned().unwrap_or(String::new())
    });

    vdb_reader.read_grid::<half::f16>(&grid_to_load).unwrap()
}
