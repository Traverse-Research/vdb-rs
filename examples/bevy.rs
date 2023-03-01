use bevy::prelude::*;
use bevy_aabb_instancing::{
    ColorOptions, ColorOptionsMap, Cuboid, Cuboids, ScalarHueColorOptions,
    VertexPullingRenderPlugin, COLOR_MODE_SCALAR_HUE,
};
use smooth_bevy_cameras::{controllers::fps::*, LookTransformPlugin};
use vdb_reader::{read_vdb, Index, Node};

use std::{error::Error, fs::File, io::BufReader};

fn main() -> Result<(), Box<dyn Error>> {
    //let f = File::open("C:/Users/Jasper/Downloads/buddha.vdb-1.0.0/buddha.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/bunny.vdb-1.0.0/bunny.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/bunny_cloud.vdb-1.0.0/bunny_cloud.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/cube.vdb-1.0.0/cube.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/crawler.vdb-1.0.0/crawler.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/dragon.vdb-1.0.0/dragon.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/emu.vdb-1.0.0/emu.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/armadillo.vdb-1.0.0/armadillo.vdb")?;

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            window: WindowDescriptor {
                title: "VDB Viewer".into(),
                ..Default::default()
            },
            ..Default::default()
        }))
        .add_plugin(VertexPullingRenderPlugin { outlines: true })
        .add_plugin(LookTransformPlugin)
        .add_plugin(FpsCameraPlugin::default())
        .add_startup_system(setup)
        .run();

    Ok(())
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut color_options_map: ResMut<ColorOptionsMap>,
) {
    let color_options_id = color_options_map.push(ColorOptions {
        scalar_hue: ScalarHueColorOptions {
            min_visible: -10000.0,
            max_visible: 10000.0,
            clamp_min: -1.0,
            clamp_max: 0.5,
            hue_zero: 240.0,
            hue_slope: -300.0,
        },
        color_mode: COLOR_MODE_SCALAR_HUE,
        wireframe: 0,
    });

    let filename = if let Some(filename) = std::env::args().skip(1).next() {
        filename
    } else {
        "C:/Users/Jasper/Downloads/armadillo.vdb-1.0.0/armadillo.vdb".to_string()
    };
    // let f = File::open("C:/Users/Jasper/Downloads/buddha.vdb-1.0.0/buddha.vdb").unwrap();
    // let f = File::open("C:/Users/Jasper/Downloads/bunny.vdb-1.0.0/bunny.vdb").unwrap();
    // let f = File::open("C:/Users/Jasper/Downloads/bunny_cloud.vdb-1.0.0/bunny_cloud.vdb").unwrap();
    // let f = File::open("C:/Users/Jasper/Downloads/cube.vdb-1.0.0/cube.vdb").unwrap();
    // let f = File::open("C:/Users/Jasper/Downloads/crawler.vdb-1.0.0/crawler.vdb").unwrap();
    // let f = File::open("C:/Users/Jasper/Downloads/dragon.vdb-1.0.0/dragon.vdb").unwrap();
    // let f = File::open("C:/Users/Jasper/Downloads/emu.vdb-1.0.0/emu.vdb").unwrap();

    let f = File::open(filename).unwrap();
    // let f = File::open("C:/Users/Jasper/Downloads/cube.vdb-1.0.0/cube.vdb").unwrap();
    // let f = File::open("C:/Users/Jasper/Downloads/explosion.vdb-1.0.0/explosion.vdb").unwrap();

    let mut reader = BufReader::new(f);

    let grid = read_vdb::<_, half::f16>(&mut reader).unwrap();

    let mesh = meshes.add(Mesh::from(shape::Cube { size: 0.01 }));
    let material = materials.add(Color::rgb(0.8, 0.7, 0.6).into());

    for root_idx in 0..grid.tree.root_nodes.len() {
        let node_5 = &grid.tree.root_nodes[root_idx];
        for idx in node_5.child_mask.iter_ones() {
            let node_4 = node_5.nodes.get(&(idx as u32)).unwrap();

            let mut instances = vec![];

            for idx in node_4.child_mask.iter_ones() {
                if true {
                    let node_3 = node_4.nodes.get(&(idx as u32)).unwrap();

                    for idx in node_3.value_mask.iter_ones() {
                        //let v = node_3.buffer[idx];
                        if let Some(v) = grid.get_value(&node_3.offset_to_global_coord(Index(idx as u32))) {
                            let global_coord = node_3.offset_to_global_coord(Index(idx as u32));

                            let c = global_coord.0.as_vec3();
                            let c = bevy::prelude::Vec3::new(c.x, c.y, c.z);
                            instances.push(Cuboid::new(
                                c * 0.01,
                                (c + bevy::prelude::Vec3::new(1.0, 1.0, 1.0)) * 0.01,
                                u32::from_le_bytes(f32::to_le_bytes(v.to_f32())),
                                // v,
                                true,
                                idx as u16,
                            ));
                        }
                    }
                } else {
                    let global_coord = node_4.offset_to_global_coord(Index(idx as u32));

                    commands.spawn(PbrBundle {
                        mesh: mesh.clone(),
                        material: material.clone(),
                        transform: Transform::from_xyz(
                            global_coord.0.x as f32 / 224.0,
                            global_coord.0.y as f32 / 224.0,
                            global_coord.0.z as f32 / 224.0,
                        ),
                        ..default()
                    });
                }
            }

            let cuboids = Cuboids::new(instances);
            let aabb = cuboids.aabb();
            commands
                .spawn(SpatialBundle::default())
                .insert((cuboids, aabb, color_options_id));
        }
    }

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
        .insert(FpsCameraBundle::new(
            FpsCameraController {
                translate_sensitivity: 2.0,
                ..Default::default()
            },
            Vec3::new(0.0, 1.0, 10.0),
            Vec3::new(0.0, 0.0, 0.0),
        ));
}

// odd / broken?
// no visuals: "C:\Users\Jasper\Downloads\smoke2.vdb-1.0.0\smoke2.vdb"
// parse error: "C:\Users\Jasper\Downloads\torus.vdb-1.0.0\torus.vdb" InvalidNodeMetadata
// parse erorr: "C:\Users\Jasper\Downloads\venusstatue.vdb-1.0.0\venusstatue.vdb" InvalidNodeMetadata
// parse error: "C:\Users\Jasper\Downloads\boat_points.vdb-1.0.0\boat_points.vdb" InvalidCompression
// parse error: "C:\Users\Jasper\Downloads\bunny_points.vdb-1.0.0\bunny_points.vdb" InvalidCompression
// parse error: "C:\Users\Jasper\Downloads\sphere_points.vdb-1.0.0\sphere_points.vdb" InvalidCompression
// parse error: "C:\Users\Jasper\Downloads\waterfall_points.vdb-1.0.0\waterfall_points.vdb" InvalidCompression
