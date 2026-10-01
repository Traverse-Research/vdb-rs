use crate::coordinates::{GlobalCoord, Index, LocalCoord};
use crate::transform::Map;
use crate::OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION;
use bitflags::bitflags;
use bitvec::prelude::*;
use bitvec::slice::IterOnes;
use glam::{IVec3, Vec3};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

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
            node_5_iter_active: Default::default(),
            node_5_iter_child: Default::default(),
            node_4_iter_active: Default::default(),
            node_4_iter_child: Default::default(),
            node_3_iter_child: Default::default(),

            node_5: None,
            node_4: None,
            node_3: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
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
    node_5_iter_active: IterOnes<'a, u64, Lsb0>,
    node_5_iter_child: IterOnes<'a, u64, Lsb0>,
    node_4_iter_active: IterOnes<'a, u64, Lsb0>,
    node_4_iter_child: IterOnes<'a, u64, Lsb0>,
    node_3_iter_child: IterOnes<'a, u64, Lsb0>,

    node_5: Option<&'a Node5<ValueTy>>,
    node_4: Option<&'a Node4<ValueTy>>,
    node_3: Option<&'a Node3<ValueTy>>,
}

impl<ValueTy> Iterator for GridIter<'_, ValueTy>
where
    ValueTy: Copy,
{
    type Item = (Vec3, ValueTy, VdbLevel);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let (Some(idx), Some(node_3)) = (self.node_3_iter_child.next(), self.node_3) {
                let v = node_3.buffer[idx];
                let global_coord = node_3.offset_to_global_coord(Index(idx as u32));
                let c = global_coord.0.as_vec3();
                return Some((c, v, VdbLevel::Voxel));
            }

            // either iterate over the child bits and continue to child, or iterate over active bits and return large voxels
            if let (Some(idx), Some(node_4)) = (self.node_4_iter_child.next(), self.node_4) {
                let node_3 = &node_4.nodes[&(idx as u32)];
                self.node_3_iter_child = node_3.value_mask.iter_ones();
                self.node_3 = Some(node_3);
                continue;
            } else if let (Some(idx), Some(node_4)) = (self.node_4_iter_active.next(), self.node_4)
            {
                return Some((
                    node_4.offset_to_global_coord(Index(idx as u32)).0.as_vec3(),
                    if self.grid.descriptor.file_version
                        < OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION
                    {
                        let node_mask_compression_idx = node_4
                            .child_mask
                            .iter()
                            .by_vals()
                            .take(idx)
                            .fold(0, |old, val| old + (!val as usize)); // count 0's before idx
                        node_4.data[node_mask_compression_idx]
                    } else {
                        node_4.data[idx]
                    },
                    VdbLevel::Node3,
                ));
            }

            // either iterate over the child bits and continue to child, or iterate over active bits and return large voxels
            if let (Some(idx), Some(node_5)) = (self.node_5_iter_child.next(), self.node_5) {
                let node_4 = &node_5.nodes[&(idx as u32)];
                self.node_4_iter_active = node_4.value_mask.iter_ones();
                self.node_4_iter_child = node_4.child_mask.iter_ones();
                self.node_4 = Some(node_4);
                continue;
            } else if let (Some(idx), Some(node_5)) = (self.node_5_iter_active.next(), self.node_5)
            {
                return Some((
                    node_5.offset_to_global_coord(Index(idx as u32)).0.as_vec3(),
                    node_5.data[idx],
                    VdbLevel::Node4,
                ));
            }

            if self.root_idx < self.grid.tree.root_nodes.len() {
                let node_5 = &self.grid.tree.root_nodes[self.root_idx];
                self.node_5_iter_active = node_5.value_mask.iter_ones();
                self.node_5_iter_child = node_5.child_mask.iter_ones();
                self.node_5 = Some(node_5);
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
    pub file_version: u32,
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
    Vec3i(IVec3),
    I32(i32),
    I64(i64),
    Float(f32),
    Bool(bool),
    Unknown { name: String, data: Vec<u8> },
}

pub trait Node {
    /// Side length of this node, in voxels.
    ///
    /// A node subdivides its own extent into `1 << LOG_2_DIM` slots per axis, and each of those
    /// slots spans `1 << TOTAL` voxels because that is what the child node underneath it covers.
    /// The side length is therefore `1 << (LOG_2_DIM + TOTAL)`, not `1 << LOG_2_DIM` -- the latter
    /// counts slots rather than voxels, and only coincides for [`Node3`], where `TOTAL` is 0.
    ///
    /// This matches OpenVDB's `DIM = 1 << TOTAL`, keeping in mind that its `TOTAL` is
    /// `Log2Dim + ChildNodeType::TOTAL` whereas ours is just `ChildNodeType::TOTAL`.
    const DIM: u32 = 1 << (Self::LOG_2_DIM + Self::TOTAL);
    /// Number of slots per axis, i.e. the number of children this node can address per axis.
    const LOG_2_DIM: u32;
    /// `LOG_2_DIM` of the child node, accumulated down to the voxel level.
    const TOTAL: u32;

    /// Number of addressable slots in this node, i.e. the exclusive upper bound on the [`Index`]
    /// returned by [`Node::local_coord_to_offset`].
    const NUM_VALUES: u32 = 1 << (3 * Self::LOG_2_DIM);

    fn local_coord_to_offset(&self, xyz: LocalCoord) -> Index {
        let index_3d = (xyz.0 & (Self::DIM - 1)) >> Self::TOTAL;
        Index((index_3d.x << (2 * Self::LOG_2_DIM)) + (index_3d.y << Self::LOG_2_DIM) + index_3d.z)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn node3() -> Node3<f32> {
        Node3 {
            buffer: Vec::new(),
            value_mask: BitVec::new(),
            origin: IVec3::ZERO,
        }
    }

    fn node4() -> Node4<f32> {
        Node4 {
            child_mask: BitVec::new(),
            value_mask: BitVec::new(),
            nodes: HashMap::new(),
            data: Vec::new(),
            origin: IVec3::ZERO,
        }
    }

    fn node5() -> Node5<f32> {
        Node5 {
            child_mask: BitVec::new(),
            value_mask: BitVec::new(),
            nodes: HashMap::new(),
            data: Vec::new(),
            origin: IVec3::ZERO,
        }
    }

    /// Every offset survives a trip out to a global coordinate and back, and the offsets cover
    /// `0..NUM_VALUES` exactly once.
    ///
    /// With the origin at zero, `offset_to_global_coord` yields the lower corner of the voxel
    /// region that the offset addresses, so feeding it back in has to reproduce the offset. This
    /// is what the old `DIM = 1 << LOG_2_DIM` broke: the mask discarded the high bits of the
    /// coordinate that `>> TOTAL` was about to shift down into place.
    fn assert_offsets_round_trip<N: Node>(node: &N) {
        let mut seen = vec![false; N::NUM_VALUES as usize];

        for offset in 0..N::NUM_VALUES {
            let global = node.offset_to_global_coord(Index(offset)).0;
            assert!(
                global.min_element() >= 0,
                "offset {offset} produced a negative coord {global:?}"
            );

            let round_tripped = node.local_coord_to_offset(LocalCoord(global.as_uvec3())).0;
            assert_eq!(
                round_tripped, offset,
                "offset {offset} round-tripped to {round_tripped} via {global:?}"
            );

            assert!(!seen[offset as usize], "offset {offset} produced twice");
            seen[offset as usize] = true;
        }

        assert!(
            seen.into_iter().all(|s| s),
            "offsets do not cover 0..{}",
            N::NUM_VALUES
        );
    }

    #[test]
    fn offsets_round_trip_through_global_coords() {
        assert_offsets_round_trip(&node3());
        assert_offsets_round_trip(&node4());
        assert_offsets_round_trip(&node5());
    }

    /// Each offset addresses a `(1 << TOTAL)` cube of voxels, so every coordinate inside that cube
    /// has to resolve back to the same offset -- not just the lower corner.
    fn assert_child_extent_is_uniform<N: Node>(node: &N) {
        let child_dim = 1 << N::TOTAL;

        for offset in 0..N::NUM_VALUES {
            let corner = node.offset_to_global_coord(Index(offset)).0.as_uvec3();

            // The interior is `child_dim^3` voxels, which is far too much to walk for `Node5`, so
            // check the corners of the cube plus its centre.
            for dx in [0, child_dim - 1] {
                for dy in [0, child_dim - 1] {
                    for dz in [0, child_dim - 1] {
                        let inside = corner + glam::UVec3::new(dx, dy, dz);
                        assert_eq!(
                            node.local_coord_to_offset(LocalCoord(inside)).0,
                            offset,
                            "{inside:?} inside child {offset} resolved elsewhere"
                        );
                    }
                }
            }

            let centre = corner + glam::UVec3::splat(child_dim / 2);
            assert_eq!(node.local_coord_to_offset(LocalCoord(centre)).0, offset);
        }
    }

    #[test]
    fn coords_within_a_child_share_its_offset() {
        assert_child_extent_is_uniform(&node3());
        assert_child_extent_is_uniform(&node4());
        assert_child_extent_is_uniform(&node5());
    }

    /// Coordinates outside the node wrap into it by the `DIM - 1` mask rather than escaping the
    /// offset range.
    fn assert_wraps_out_of_range<N: Node>(node: &N) {
        for offset in 0..N::NUM_VALUES {
            let corner = node.offset_to_global_coord(Index(offset)).0.as_uvec3();
            let wrapped = corner + glam::UVec3::new(N::DIM, 3 * N::DIM, 7 * N::DIM);

            assert_eq!(
                node.local_coord_to_offset(LocalCoord(wrapped)).0,
                offset,
                "{wrapped:?} did not wrap back onto offset {offset}"
            );
        }
    }

    #[test]
    fn out_of_range_coords_wrap_into_the_node() {
        assert_wraps_out_of_range(&node3());
        assert_wraps_out_of_range(&node4());
        assert_wraps_out_of_range(&node5());
    }

    /// Guards the constants themselves, since `DIM` is what regressed. The voxel extents are the
    /// 8/128/4096 of OpenVDB's `LeafNode<_, 3>`, `InternalNode<_, 4>` and `InternalNode<_, 5>`.
    #[test]
    fn node_extents_match_openvdb() {
        assert_eq!(<Node3<f32> as Node>::DIM, 8);
        assert_eq!(<Node4<f32> as Node>::DIM, 128);
        assert_eq!(<Node5<f32> as Node>::DIM, 4096);

        assert_eq!(<Node3<f32> as Node>::NUM_VALUES, 512);
        assert_eq!(<Node4<f32> as Node>::NUM_VALUES, 4096);
        assert_eq!(<Node5<f32> as Node>::NUM_VALUES, 32768);
    }
}
