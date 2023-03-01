#[derive(Debug)]
pub struct GlobalCoord(pub glam::IVec3);

#[derive(Debug)]
pub struct LocalCoord(pub glam::UVec3);

#[derive(Debug)]
pub struct Index(pub u32);

#[derive(Debug, Eq, PartialEq)]
pub struct TileKey(pub u64);

pub fn local_coord_to_tile_key(ijk: &LocalCoord) -> TileKey {
    let iu = ijk.0.x as u64 >> 12;
    let ju = ijk.0.y as u64 >> 12;
    let ku = ijk.0.z as u64 >> 12;
    TileKey((ku) | (ju << 21) | (iu << 42))
}