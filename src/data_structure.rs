use crate::coordinates::{GlobalCoord, Index, LocalCoord};
use crate::transform::Map;
use bitflags::bitflags;
use bitvec::prelude::*;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug)]
pub struct Grid<ValueTy> {
    pub tree: Tree<ValueTy>,
    pub transform: Map,
    pub grid_descriptor: GridDescriptor,
}

#[derive(Debug)]
pub struct GridDescriptor {
    pub name: String,
    pub grid_type: String,
    pub instance_parent: String,
    pub grid_pos: u64,
    pub block_pos: u64,
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
}

#[derive(Debug, Default)]
pub struct Metadata(pub HashMap<String, MetadataValue>);

impl Metadata {
    pub fn is_half_float(&self) -> bool {
        self.0.get("is_saved_as_half_float") == Some(&MetadataValue::Bool(true))
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum MetadataValue {
    String(String),
    Vec3i(glam::IVec3),
    I64(i64),
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
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
    pub struct Compression: u32 {
        const NONE = 0;
        const ZIP = 0x1;
        const ACTIVE_MASK = 0x2;
        const BLOSC = 0x4;
        const DEFAULT_COMPRESSION = Self::BLOSC.bits | Self::ACTIVE_MASK.bits;
    }
}

#[derive(Debug)]
pub struct ArchiveHeader {
    pub file_version: u32,
    pub library_version_major: u32,
    pub library_version_minor: u32,
    pub has_grid_offsets: bool,
    pub compression: Compression,
    pub guid: String,
    pub meta_data: Metadata,
    pub grid_count: u32,
}
