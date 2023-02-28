#[derive(Debug)]
pub enum Map {
    UniformScaleMap {
        scale_values: glam::DVec3,
        voxel_size: glam::DVec3,
        scale_values_inverse: glam::DVec3,
        inv_scale_sqr: glam::DVec3,
        inv_twice_scale: glam::DVec3,
    },
    UniformScaleTranslateMap {
        translation: glam::DVec3,
        scale_values: glam::DVec3,
        voxel_size: glam::DVec3,
        scale_values_inverse: glam::DVec3,
        inv_scale_sqr: glam::DVec3,
        inv_twice_scale: glam::DVec3,
    }
}
