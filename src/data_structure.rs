use crate::coordinates::{GlobalCoord, Index, LocalCoord};
use crate::transform::Map;
use bitflags::bitflags;
use bitvec::prelude::*;
use bitvec::slice::IterOnes;
use glam::Vec3;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug)]
pub struct Grid<ValueTy> {
    pub tree: Tree<ValueTy>,
    pub transform: Map,
    pub grid_descriptor: GridDescriptor,
}

impl<ValueTy> Grid<ValueTy> {
    pub fn iter(&self) -> GridIter<'_, ValueTy> {
        GridIter {
            grid: self,
            root_idx: 0,
            node_5_iter: Default::default(),
            node_4_iter: Default::default(),
            node_3_iter: Default::default(),

            node_5: None,
            node_4: None,
            node_3: None,
        }
    }
}

pub struct GridIter<'a, ValueTy> {
    grid: &'a Grid<ValueTy>,
    root_idx: usize,
    node_5_iter: IterOnes<'a, u64, Lsb0>,
    node_4_iter: IterOnes<'a, u64, Lsb0>,
    node_3_iter: IterOnes<'a, u64, Lsb0>,

    node_5: Option<&'a Node5<ValueTy>>,
    node_4: Option<&'a Node4<ValueTy>>,
    node_3: Option<&'a Node3<ValueTy>>,
}

impl<'a, ValueTy> Iterator for GridIter<'a, ValueTy>
where
    ValueTy: Copy,
{
    type Item = (Vec3, ValueTy);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let (Some(idx), Some(node_3)) = (self.node_3_iter.next(), self.node_3) {
                let v = node_3.buffer[idx];
                let global_coord = node_3.offset_to_global_coord(Index(idx as u32));
                let c = global_coord.0.as_vec3();
                return Some((c, v));
            }
            if let (Some(idx), Some(node_4)) = (self.node_4_iter.next(), self.node_4) {
                let node_3 = &node_4.nodes[&(idx as u32)];
                self.node_3_iter = node_3.value_mask.iter_ones();
                self.node_3 = Some(node_3);
                continue;
            }
            if let (Some(idx), Some(node_5)) = (self.node_5_iter.next(), self.node_5) {
                let node_4 = &node_5.nodes[&(idx as u32)];
                self.node_4_iter = node_4.child_mask.iter_ones();
                self.node_4 = Some(node_4);
                continue;
            }
            if self.root_idx < self.grid.tree.root_nodes.len() {
                let node_5 = &self.grid.tree.root_nodes[self.root_idx];
                self.node_5_iter = node_5.child_mask.iter_ones();
                self.node_5 = Some(node_5);
                self.root_idx += 1;
                continue;
            }
            return None;
        }
    }
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
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Compression: u32 {
        const NONE = 0;
        const ZIP = 0x1;
        const ACTIVE_MASK = 0x2;
        const BLOSC = 0x4;
        const DEFAULT_COMPRESSION = Self::BLOSC.bits() | Self::ACTIVE_MASK.bits();
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
