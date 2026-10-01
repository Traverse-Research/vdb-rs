//! A small orbit camera for the examples.
//!
//! `smooth-bevy-cameras` has no release supporting bevy 0.19, so the examples carry their own
//! controller rather than holding the whole dependency stack back.
//!
//! Left-drag orbits, right-drag pans, the scroll wheel zooms.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;

pub struct OrbitCameraPlugin;

impl Plugin for OrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, run_orbit_camera);
    }
}

#[derive(Component)]
pub struct OrbitCamera {
    pub focus: Vec3,
    pub radius: f32,
    pub sensitivity: f32,
    yaw: f32,
    pitch: f32,
}

impl OrbitCamera {
    /// Builds a camera orbiting `focus`, placed at `eye`, along with the matching [`Transform`].
    pub fn new(eye: Vec3, focus: Vec3) -> (Self, Transform) {
        let offset = eye - focus;
        let radius = offset.length().max(f32::EPSILON);
        let pitch = (offset.y / radius).clamp(-1.0, 1.0).asin();
        let yaw = offset.x.atan2(offset.z);

        let camera = Self {
            focus,
            radius,
            sensitivity: 0.005,
            yaw,
            pitch,
        };
        let transform = camera.transform();
        (camera, transform)
    }

    fn transform(&self) -> Transform {
        let offset = Vec3::new(
            self.radius * self.pitch.cos() * self.yaw.sin(),
            self.radius * self.pitch.sin(),
            self.radius * self.pitch.cos() * self.yaw.cos(),
        );
        Transform::from_translation(self.focus + offset).looking_at(self.focus, Vec3::Y)
    }
}

fn run_orbit_camera(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    mut camera: Query<(&mut OrbitCamera, &mut Transform)>,
) {
    let Ok((mut orbit, mut transform)) = camera.single_mut() else {
        return;
    };

    let mut changed = false;

    let delta = mouse_motion.delta;
    if delta != Vec2::ZERO && mouse_buttons.pressed(MouseButton::Left) {
        orbit.yaw -= delta.x * orbit.sensitivity;
        let limit = core::f32::consts::FRAC_PI_2 - 0.01;
        orbit.pitch = (orbit.pitch + delta.y * orbit.sensitivity).clamp(-limit, limit);
        changed = true;
    }

    if delta != Vec2::ZERO && mouse_buttons.pressed(MouseButton::Right) {
        // Pan across the view plane, scaled by distance so it feels the same at any zoom.
        let right = *transform.right();
        let up = *transform.up();
        let pan = (-right * delta.x + up * delta.y) * orbit.radius * 0.001;
        orbit.focus += pan;
        changed = true;
    }

    if mouse_scroll.delta.y != 0.0 {
        orbit.radius = (orbit.radius * (1.0 - mouse_scroll.delta.y * 0.1)).max(0.05);
        changed = true;
    }

    if changed {
        *transform = orbit.transform();
    }
}
