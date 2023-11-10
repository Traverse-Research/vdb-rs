use crate::coordinates::{GlobalCoord, Index, LocalCoord};
use crate::transform::Map;
use crate::OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION;
use bitflags::bitflags;
use bitvec::index::BitMask;
use bitvec::prelude::*;
use bitvec::slice::IterOnes;
use glam::{IVec3, Vec3};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::ops::AddAssign;

#[derive(thiserror::Error, Debug)]
pub enum GridMetadataError {
    #[error("Field {0} not in grid metadata")]
    FieldNotPresent(String),
}

#[derive(Debug)]
pub struct Grid<ValueTy> {
    pub tree: Tree<ValueTy>,
    pub transform: Map,
    pub descriptor: GridDescriptor,
}

impl<ValueTy> Grid<ValueTy> {
    pub fn iter(&self) -> GridIter<'_, ValueTy> {
        GridIter {
            grid: self,
            root_idx: 0,
            node_5_bit_index: 1 << 5,
            node_4_bit_index: 1 << 4,
            node_3_bit_index: 1 << 3,

            node_5: None,
            node_4: None,
            node_3: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VdbLevel {
    Node4,
    Node3,
    Voxel,
}
impl VdbLevel {
    pub fn scale(self) -> f32 {
        match self {
            VdbLevel::Node4 => (1 << (4 + 3)) as f32,
            VdbLevel::Node3 => (1 << 3) as f32,
            VdbLevel::Voxel => 1.0,
        }
    }
}

pub struct GridIter<'a, ValueTy> {
    grid: &'a Grid<ValueTy>,
    root_idx: usize,
    node_5_bit_index: usize,
    node_4_bit_index: usize,
    node_3_bit_index: usize,

    node_5: Option<&'a Node5<ValueTy>>,
    node_4: Option<&'a Node4<ValueTy>>,
    node_3: Option<&'a Node3<ValueTy>>,
}

impl<'a, ValueTy> GridIter<'a, ValueTy>
where
    ValueTy: Copy,
{
    fn get_tile_value(
        &self,
        bit_index: usize,
        bit_mask: &BitVec<u64>,
        data: &Vec<ValueTy>,
    ) -> ValueTy {
        if self.grid.descriptor.file_version < OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION {
            let node_mask_compression_idx = bit_mask
                .iter()
                .by_vals()
                .take(bit_index)
                .fold(0, |old, val| old + (!val as usize)); // count 0's before idx
            data[node_mask_compression_idx]
        } else {
            data[bit_index]
        }
    }
}

impl<'a, ValueTy> Iterator for GridIter<'a, ValueTy>
where
    ValueTy: Copy,
{
    type Item = (Vec3, ValueTy, VdbLevel);

    fn next(&mut self) -> Option<Self::Item> {
        'outer: loop {
            if let Some(node_3) = self.node_3 {
                while self.node_3_bit_index < (1 << 3 * 3) {
                    let idx = self.node_3_bit_index;
                    self.node_3_bit_index += 1;
                    if node_3.value_mask[idx] {
                        let v = node_3.buffer[idx];
                        let global_coord = node_3.offset_to_global_coord(Index(idx as u32));
                        let c = global_coord.0.as_vec3();
                        return Some((c, v, VdbLevel::Voxel));
                    }
                }
            }

            // either iterate over the child bits and continue to child, or iterate over active bits and return large voxels
            if let Some(node_4) = self.node_4 {
                while self.node_4_bit_index < (1 << 4 * 3) {
                    let idx = self.node_4_bit_index;
                    self.node_4_bit_index += 1;
                    if node_4.value_mask[idx] {
                        self.node_4_bit_index = (1 << 4 * 3);
                        return Some((
                            node_4.offset_to_global_coord(Index(idx as u32)).0.as_vec3(),
                            self.get_tile_value(idx, &node_4.child_mask, &node_4.data),
                            VdbLevel::Node3,
                        ));
                    } else if node_4.child_mask[idx] {
                        self.node_3_bit_index = 0;
                        self.node_3 = Some(&node_4.nodes[&(idx as u32)]);
                        continue 'outer;
                    }
                }
            }

            if let Some(node_5) = self.node_5 {
                while self.node_5_bit_index < (1 << 5 * 3) {
                    let idx = self.node_5_bit_index;
                    self.node_5_bit_index += 1;
                    if node_5.value_mask[idx] {
                        self.node_5_bit_index = (1 << 5 * 3);
                        return Some((
                            node_5.offset_to_global_coord(Index(idx as u32)).0.as_vec3(),
                            self.get_tile_value(idx, &node_5.child_mask, &node_5.data),
                            VdbLevel::Node4,
                        ));
                    } else if node_5.child_mask[idx] {
                        self.node_4_bit_index = 0;
                        self.node_4 = Some(&node_5.nodes[&(idx as u32)]);
                        continue 'outer;
                    }
                }
            }

            if self.root_idx < self.grid.tree.root_nodes.len() {
                self.node_5_bit_index = 0;
                self.node_5 = Some(&self.grid.tree.root_nodes[self.root_idx]);
                self.root_idx += 1;
                continue;
            }
            return None;
        }
    }
}

#[derive(Debug, Clone)]
pub struct GridDescriptor {
    pub name: String,
    pub(crate) file_version: u32,
    /// If not empty, the name of another grid that shares this grid's tree
    pub instance_parent: String,
    pub grid_type: String,
    /// Location in the stream where the grid data is stored
    pub grid_pos: u64,
    /// Location in the stream where the grid blocks are stored
    pub block_pos: u64,
    /// Location in the stream where the next grid descriptor begins
    pub end_pos: u64,
    pub compression: Compression,
    pub meta_data: Metadata,
}

impl GridDescriptor {
    pub(crate) fn seek_to_grid<R: Read + Seek>(
        &self,
        reader: &mut R,
    ) -> Result<u64, std::io::Error> {
        reader.seek(SeekFrom::Start(self.grid_pos))
    }

    pub(crate) fn seek_to_blocks<R: Read + Seek>(
        &self,
        reader: &mut R,
    ) -> Result<u64, std::io::Error> {
        reader.seek(SeekFrom::Start(self.block_pos))
    }

    // below values should always be present, see https://github.com/AcademySoftwareFoundation/openvdb/blob/master/openvdb/openvdb/Grid.cc#L387
    pub fn aabb_min(&self) -> Result<IVec3, GridMetadataError> {
        match self.meta_data.0["file_bbox_min"] {
            MetadataValue::Vec3i(v) => Ok(v),
            _ => Err(GridMetadataError::FieldNotPresent(
                "file_bbox_min".to_string(),
            )),
        }
    }
    pub fn aabb_max(&self) -> Result<IVec3, GridMetadataError> {
        match self.meta_data.0["file_bbox_max"] {
            MetadataValue::Vec3i(v) => Ok(v),
            _ => Err(GridMetadataError::FieldNotPresent(
                "file_bbox_max".to_string(),
            )),
        }
    }
    pub fn mem_bytes(&self) -> Result<i64, GridMetadataError> {
        match self.meta_data.0["file_mem_bytes"] {
            MetadataValue::I64(v) => Ok(v),
            _ => Err(GridMetadataError::FieldNotPresent(
                "file_mem_bytes".to_string(),
            )),
        }
    }
    pub fn voxel_count(&self) -> Result<i64, GridMetadataError> {
        match self.meta_data.0["file_voxel_count"] {
            MetadataValue::I64(v) => Ok(v),
            _ => Err(GridMetadataError::FieldNotPresent(
                "file_voxel_count".to_string(),
            )),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Metadata(pub HashMap<String, MetadataValue>);

impl Metadata {
    pub fn is_half_float(&self) -> bool {
        self.0.get("is_saved_as_half_float") == Some(&MetadataValue::Bool(true))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    String(String),
    Vec3i(glam::IVec3),
    I32(i32),
    I64(i64),
    Float(f32),
    Bool(bool),
    Unknown { name: String, data: Vec<u8> },
}

pub trait Node {
    const DIM: u32 = 1 << Self::LOG_2_DIM;
    const LOG_2_DIM: u32;
    const TOTAL: u32;

    fn local_coord_to_offset(&self, xyz: LocalCoord) -> Index {
        Index(
            (((xyz.0[0] & (Self::DIM - 1)) >> Self::TOTAL) << (2 * Self::LOG_2_DIM))
                + (((xyz.0[1] & (Self::DIM - 1)) >> Self::TOTAL) << Self::LOG_2_DIM)
                + ((xyz.0[2] & (Self::DIM - 1)) >> Self::TOTAL),
        )
    }

    fn offset_to_local_coord(&self, offset: Index) -> LocalCoord {
        assert!(
            offset.0 < (1 << (3 * Self::LOG_2_DIM)),
            "Offset {} out of bounds",
            offset.0
        );

        let x = offset.0 >> (2 * Self::LOG_2_DIM);
        let offset = offset.0 & ((1 << (2 * Self::LOG_2_DIM)) - 1);

        let y = offset >> Self::LOG_2_DIM;
        let z = offset & ((1 << Self::LOG_2_DIM) - 1);

        LocalCoord(glam::UVec3::new(x, y, z))
    }

    fn offset_to_global_coord(&self, offset: Index) -> GlobalCoord {
        let mut local_coord = self.offset_to_local_coord(offset);
        local_coord.0[0] <<= Self::TOTAL;
        local_coord.0[1] <<= Self::TOTAL;
        local_coord.0[2] <<= Self::TOTAL;
        GlobalCoord(local_coord.0.as_ivec3() + self.offset())
    }

    fn offset(&self) -> glam::IVec3;
}

#[derive(Debug)]
pub struct NodeHeader<ValueTy> {
    pub child_mask: BitVec<u64, Lsb0>,
    pub value_mask: BitVec<u64, Lsb0>,
    pub data: Vec<ValueTy>,
    pub log_2_dim: u32,
}

#[derive(Debug)]
pub struct Node3<ValueTy> {
    pub buffer: Vec<ValueTy>,
    pub value_mask: BitVec<u64, Lsb0>,
    pub origin: glam::IVec3,
}

impl<ValueTy> Node for Node3<ValueTy> {
    const LOG_2_DIM: u32 = 3;
    const TOTAL: u32 = 0;

    fn offset(&self) -> glam::IVec3 {
        self.origin
    }
}

#[derive(Debug)]
pub struct Node4<ValueTy> {
    pub child_mask: BitVec<u64, Lsb0>,
    pub value_mask: BitVec<u64, Lsb0>,
    pub nodes: HashMap<u32, Node3<ValueTy>>,
    pub data: Vec<ValueTy>,
    pub origin: glam::IVec3,
}

impl<ValueTy> Node for Node4<ValueTy> {
    const LOG_2_DIM: u32 = 4;
    const TOTAL: u32 = 3;

    fn offset(&self) -> glam::IVec3 {
        self.origin
    }
}

#[derive(Debug)]
pub struct Node5<ValueTy> {
    pub child_mask: BitVec<u64, Lsb0>,
    pub value_mask: BitVec<u64, Lsb0>,
    pub nodes: HashMap<u32, Node4<ValueTy>>,
    pub data: Vec<ValueTy>,
    pub origin: glam::IVec3,
}

impl<ValueTy> Node for Node5<ValueTy> {
    const LOG_2_DIM: u32 = 5;
    const TOTAL: u32 = 7;

    fn offset(&self) -> glam::IVec3 {
        self.origin
    }
}

#[derive(Debug)]
pub struct Tree<ValueTy> {
    pub root_nodes: Vec<Node5<ValueTy>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeMetaData {
    NoMaskOrInactiveVals,
    NoMaskAndMinusBg,
    NoMaskAndOneInactiveVal,
    MaskAndNoInactiveVals,
    MaskAndOneInactiveVal,
    MaskAndTwoInactiveVals,
    NoMaskAndAllVals,
}

bitflags! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Compression: u32 {
        const NONE = 0;
        const ZIP = 0x1;
        const ACTIVE_MASK = 0x2;
        const BLOSC = 0x4;
        const DEFAULT_COMPRESSION = Self::BLOSC.bits() | Self::ACTIVE_MASK.bits();
    }
}

#[derive(Debug, Default)]
pub struct ArchiveHeader {
    /// The version of the file that was read
    pub file_version: u32,
    /// The version of the library that was used to create the file that was read
    pub library_version_major: u32,
    pub library_version_minor: u32,
    /// Unique tag, a random 16-byte (128-bit) value, stored as a string format.
    pub guid: String,
    /// Flag indicating whether the input stream contains grid offsets and therefore supports partial reading
    pub has_grid_offsets: bool,
    /// Flags indicating whether and how the data stream is compressed
    pub compression: Compression,
    /// the number of grids on the input stream
    pub grid_count: u32,
    /// The metadata for the input stream
    pub meta_data: Metadata,
}
