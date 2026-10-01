//! The fly camera shared by the examples.
//!
//! `smooth-bevy-cameras` has no release supporting bevy 0.19, so the examples carry their own
//! controller rather than holding the whole dependency stack back.  It keeps that crate's
//! sensitivities.
//!
//! - drag with the left (or right) button to look around
//! - `WASD` to move, `Q`/`E` down and up, shift to go faster
//! - scroll to move along the view, or to set the movement speed while dragging
//! - drag with both buttons, or the middle one, to pan
//!
//! Only a held button turns the camera, so the pointer stays free for the UI, and the cursor is
//! locked for the duration of a drag so it cannot run out of screen.  Left-drag alone is enough
//! to look around, because a trackpad has no middle button and an awkward right one.

use bevy::ecs::system::SystemParam;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use bevy_egui::input::EguiWantsInput;

/// Straight up and down are singular for a yaw/pitch camera, so stop just short of them.
const PITCH_LIMIT: f32 = core::f32::consts::FRAC_PI_2 - 0.01;

/// Radians of rotation per pixel of mouse movement.
const ROTATE_SENSITIVITY: f32 = 0.2 / 60.0;
/// Units of movement per pixel of mouse movement.
const TRANSLATE_SENSITIVITY: f32 = 2.0 / 60.0;
/// Units of movement per line of scroll wheel.
const WHEEL_TRANSLATE_SENSITIVITY: f32 = 50.0 / 60.0;
/// Units per second of [`FlyCamera::speed`] gained per line of scroll wheel.
const WHEEL_SPEED_SENSITIVITY: f32 = 5.0;
/// How much faster the camera moves while shift is held.
const BOOST: f32 = 4.0;
/// The mouse buttons that drive the camera.
const DRAG_BUTTONS: [MouseButton; 3] = [MouseButton::Left, MouseButton::Middle, MouseButton::Right];

/// Adds the [`FlyCamera`] controller.
pub struct FlyCameraPlugin;

impl Plugin for FlyCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, run_fly_camera);
    }
}

/// A camera that flies through the scene, driven like Unreal Engine's viewport.
///
/// Right-drag looks around, left-drag turns while moving along the view, and dragging with the
/// middle button (or both outer buttons) pans.  `W`/`A`/`S`/`D` move, `Q`/`E` descend and ascend,
/// and holding shift speeds all of that up.  The scroll wheel moves forwards and backwards, or
/// sets [`FlyCamera::speed`] while a mouse button is held.
#[derive(Component)]
pub struct FlyCamera {
    /// Keyboard movement speed, in units per second.
    pub speed: f32,
    yaw: f32,
    pitch: f32,
}

impl FlyCamera {
    /// Builds a camera at `eye` looking at `target`, along with the matching [`Transform`].
    pub fn new(eye: Vec3, target: Vec3) -> (Self, Transform) {
        let forward = (target - eye).normalize_or_zero();
        let camera = Self {
            speed: 10.0,
            yaw: (-forward.x).atan2(-forward.z),
            pitch: forward.y.clamp(-1.0, 1.0).asin(),
        };
        let transform = Transform::from_translation(eye).looking_to(camera.forward(), Vec3::Y);
        (camera, transform)
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            -self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            -self.pitch.cos() * self.yaw.cos(),
        )
    }
}

/// The mouse state the camera reads, plus the bookkeeping that keeps it out of the UI's way.
#[derive(SystemParam)]
struct Mouse<'w, 's> {
    buttons: Res<'w, ButtonInput<MouseButton>>,
    motion: Res<'w, AccumulatedMouseMotion>,
    scroll: Res<'w, AccumulatedMouseScroll>,
    /// Only the examples that add `EguiPlugin` have this resource; without a UI nothing else is
    /// interested in the mouse.
    egui_wants_input: Option<Res<'w, EguiWantsInput>>,
    owned_by_camera: Local<'s, bool>,
}

impl Mouse<'_, '_> {
    /// Egui and bevy both see every mouse message, and egui does not consume the ones it acts on,
    /// so without this a drag on the settings window would move the camera too.  Who owns the
    /// pointer is decided while no button is held and then kept for the rest of the drag, so that
    /// a camera move wandering over a panel is not cut short halfway.
    fn claim(&mut self) -> bool {
        if !self.buttons.any_pressed(DRAG_BUTTONS) {
            *self.owned_by_camera = self
                .egui_wants_input
                .as_ref()
                .is_none_or(|egui| !egui.wants_any_pointer_input());
        }
        *self.owned_by_camera
    }
}

fn run_fly_camera(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mouse: Mouse,
    mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>,
    mut camera: Query<(&mut FlyCamera, &mut Transform)>,
) {
    let Ok((mut fly, mut transform)) = camera.single_mut() else {
        return;
    };
    let claimed = mouse.claim();

    // A drag must not run out of screen: once the pointer reaches the edge of the display the
    // system stops reporting motion in that direction, so the camera stops turning and dragging
    // back merely unwinds it.  Locking parks the pointer for the duration of the drag, which both
    // keeps the relative motion coming and hides a cursor that has nothing to point at.
    let grabbed = claimed && mouse.buttons.any_pressed(DRAG_BUTTONS);
    if let Ok(mut cursor) = cursor.single_mut() {
        let grab_mode = if grabbed {
            // macOS falls back to `None` for `Confined`, which would not help here.
            CursorGrabMode::Locked
        } else {
            CursorGrabMode::None
        };
        if cursor.grab_mode != grab_mode {
            cursor.grab_mode = grab_mode;
            cursor.visible = !grabbed;
        }
    }

    // The keys keep working wherever the pointer happens to rest; only the mouse is shared.
    let left = claimed && mouse.buttons.pressed(MouseButton::Left);
    let middle = claimed && mouse.buttons.pressed(MouseButton::Middle);
    let right = claimed && mouse.buttons.pressed(MouseButton::Right);
    // Both outer buttons together pan, like the middle one, rather than turning.  Everything else
    // held turns.  Note a trackpad has no middle button and an awkward right one, so plain
    // left-drag has to be enough on its own to look around.
    let panning = middle || (left && right);
    let looking = (left || right) && !panning;
    // Only a held button drives the camera: the pointer has to be free to reach the UI, and an
    // ungrabbed drag would run into the edge of the display.
    let delta = if claimed && (looking || panning) {
        mouse.motion.delta
    } else {
        Vec2::ZERO
    };

    let mut rotated = false;
    if delta.x != 0.0 && looking {
        fly.yaw -= delta.x * ROTATE_SENSITIVITY;
        rotated = true;
    }
    if delta.y != 0.0 && looking {
        fly.pitch = (fly.pitch - delta.y * ROTATE_SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        rotated = true;
    }

    let forward = fly.forward();
    let right_axis = forward.cross(Vec3::Y).normalize_or_zero();
    let mut translation = Vec3::ZERO;

    // While dragging, the wheel sets the movement speed; on its own it moves along the view.
    let scroll = if claimed { mouse.scroll.delta.y } else { 0.0 };
    if scroll != 0.0 {
        if left || middle || right {
            fly.speed = (fly.speed + scroll * WHEEL_SPEED_SENSITIVITY).max(0.01);
        } else {
            translation += forward * scroll * WHEEL_TRANSLATE_SENSITIVITY;
        }
    }

    if delta != Vec2::ZERO && panning {
        translation += (-right_axis * delta.x + Vec3::Y * delta.y) * TRANSLATE_SENSITIVITY;
    }

    let mut direction = Vec3::ZERO;
    for (key, axis) in [
        (KeyCode::KeyW, forward),
        (KeyCode::KeyS, -forward),
        (KeyCode::KeyD, right_axis),
        (KeyCode::KeyA, -right_axis),
        (KeyCode::KeyE, Vec3::Y),
        (KeyCode::KeyQ, Vec3::NEG_Y),
    ] {
        if keyboard.pressed(key) {
            direction += axis;
        }
    }
    let boost = if keyboard.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
        BOOST
    } else {
        1.0
    };
    translation += direction.normalize_or_zero() * fly.speed * boost * time.delta_secs();

    if rotated || translation != Vec3::ZERO {
        *transform = Transform::from_translation(transform.translation + translation)
            .looking_to(forward, Vec3::Y);
    }
}
