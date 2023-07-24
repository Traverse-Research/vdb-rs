use crate::coordinates::Index;
use crate::data_structure::{
    ArchiveHeader, Compression, Grid, GridDescriptor, Metadata, MetadataValue, Node, Node3, Node4,
    Node5, NodeHeader, NodeMetaData, Tree,
};
use crate::transform::Map;
use crate::LeafDataType;

use bitvec::prelude::*;
use blosc_src::blosc_cbuffer_sizes;
use bytemuck::{bytes_of_mut, cast_slice_mut, Pod, Zeroable};
use byteorder::{LittleEndian, ReadBytesExt};

use half::f16;
use log::{trace, warn};
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom};

pub const OPENVDB_MIN_SUPPORTED_VERSION: u32 = OPENVDB_FILE_VERSION_ROOTNODE_MAP;

pub const OPENVDB_FILE_VERSION_ROOTNODE_MAP: u32 = 213;
pub const OPENVDB_FILE_VERSION_INTERNALNODE_COMPRESSION: u32 = 214;
pub const OPENVDB_FILE_VERSION_SIMPLIFIED_GRID_TYPENAME: u32 = 215;
pub const OPENVDB_FILE_VERSION_GRID_INSTANCING: u32 = 216;
pub const OPENVDB_FILE_VERSION_BOOL_LEAF_OPTIMIZATION: u32 = 217;
pub const OPENVDB_FILE_VERSION_BOOST_UUID: u32 = 218;
pub const OPENVDB_FILE_VERSION_NO_GRIDMAP: u32 = 219;
pub const OPENVDB_FILE_VERSION_NEW_TRANSFORM: u32 = 219;
pub const OPENVDB_FILE_VERSION_SELECTIVE_COMPRESSION: u32 = 220;
pub const OPENVDB_FILE_VERSION_FLOAT_FRUSTUM_BBOX: u32 = 221;
pub const OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION: u32 = 222;
pub const OPENVDB_FILE_VERSION_BLOSC_COMPRESSION: u32 = 223;
pub const OPENVDB_FILE_VERSION_POINT_INDEX_GRID: u32 = 223;
pub const OPENVDB_FILE_VERSION_MULTIPASS_IO: u32 = 224;

#[derive(thiserror::Error, Debug)]
pub enum ParseError {
    #[error("Magic bytes mismatched")]
    MagicMismatch,
    #[error("Unsupported VDB version")]
    UnsupportedVersion(u32),
    #[error("Invalid compression {0}")]
    InvalidCompression(u32),
    #[error("Invalid node meta-data: {0}")]
    InvalidNodeMetadata(u8),
    #[error("Invalid Blosc data")]
    InvalidBloscData,
    #[error("Unsupported Blosc format")]
    UnsupportedBloscFormat,
    #[error("Invalid grid name: {0}.")]
    InvalidGridName(String),
    #[error("IoError")]
    IoError(#[from] std::io::Error),
}

fn read_string<R: Read + Seek>(reader: &mut R, len: usize) -> Result<String, ParseError> {
    let mut string = String::with_capacity(len);
    for _ in 0..len {
        let c = reader.read_u8()? as char;
        string.push(c);
    }
    Ok(string)
}

fn read_d_vec3<R: Read + Seek>(reader: &mut R) -> Result<glam::DVec3, ParseError> {
    let x = reader.read_f64::<LittleEndian>()?;
    let y = reader.read_f64::<LittleEndian>()?;
    let z = reader.read_f64::<LittleEndian>()?;
    Ok(glam::DVec3::new(x, y, z))
}

fn read_i_vec3<R: Read + Seek>(reader: &mut R) -> Result<glam::IVec3, ParseError> {
    let x = reader.read_i32::<LittleEndian>()?;
    let y = reader.read_i32::<LittleEndian>()?;
    let z = reader.read_i32::<LittleEndian>()?;

    Ok(glam::IVec3::new(x, y, z))
}

#[derive(Debug)]
pub struct VdbReader<R: Read + Seek> {
    reader: R,
    pub header: ArchiveHeader,
    pub grid_descriptors: BTreeMap<String, GridDescriptor>,
}

impl<R: Read + Seek> VdbReader<R> {
    pub fn new(mut reader: R) -> Result<Self, ParseError> {
        let magic = reader.read_u64::<LittleEndian>()?;
        if magic == 0x2042445600000000 {
            return Err(ParseError::MagicMismatch);
        }

        let file_version = reader.read_u32::<LittleEndian>()?;
        if file_version < OPENVDB_MIN_SUPPORTED_VERSION {
            return Err(ParseError::UnsupportedVersion(file_version));
        }

        // Stored from version 211 onward, our minimum supported version is 213
        let library_version_major = reader.read_u32::<LittleEndian>()?;
        let library_version_minor = reader.read_u32::<LittleEndian>()?;

        // Stored from version 212 onward, our minimum supported version is 213
        let has_grid_offsets = reader.read_u8()? == 1;

        // From version 222 on, compression information is stored per grid.
        let mut compression = Compression::DEFAULT_COMPRESSION;
        if file_version < OPENVDB_FILE_VERSION_BLOSC_COMPRESSION {
            // Prior to the introduction of Blosc, ZLIB was the default compression scheme.
            compression = Compression::ZIP | Compression::ACTIVE_MASK;
        }

        // [range_start, range_end)
        if (OPENVDB_FILE_VERSION_SELECTIVE_COMPRESSION..OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION)
            .contains(&file_version)
        {
            let is_compressed = reader.read_u8()? == 1;
            if is_compressed {
                compression = Compression::ZIP;
            } else {
                compression = Compression::NONE;
            }
        }

        let guid = if file_version >= OPENVDB_FILE_VERSION_BOOST_UUID {
            // UUID is stored as fixed-length ASCII string
            // The extra 4 bytes are for the hyphens.
            read_string(&mut reader, 36)?
        } else {
            // Older versions stored the UUID as a byte string.
            todo!("File version {}", file_version);
        };

        let meta_data = Self::read_metadata(&mut reader)?;
        let grid_count = reader.read_u32::<LittleEndian>()?;

        let header = ArchiveHeader {
            file_version,
            library_version_major,
            library_version_minor,
            has_grid_offsets,
            compression,
            guid,
            meta_data,
            grid_count,
        };

        let grid_descriptors = Self::read_grid_descriptors(&header, &mut reader)?;

        Ok(Self {
            reader,
            header,
            grid_descriptors,
        })
    }

    pub fn read_grid(&mut self, name: &str) -> Result<Grid, ParseError> {
        let grid_descriptor = self.grid_descriptors.get(name).cloned();
        let gd = grid_descriptor.ok_or_else(|| ParseError::InvalidGridName(name.to_owned()))?;
        Self::read_grid_internal(&self.header, &mut self.reader, gd)
    }

    pub fn available_grids(&self) -> Vec<String> {
        self.grid_descriptors.keys().cloned().collect()
    }

    fn read_name(reader: &mut R) -> Result<String, ParseError> {
        let len = reader.read_u32::<LittleEndian>()? as usize;
        read_string(reader, len)
    }

    fn read_transform(reader: &mut R) -> Result<Map, ParseError> {
        let name = Self::read_name(reader)?;

        Ok(match name.as_str() {
            "UniformScaleMap" => Map::UniformScaleMap {
                scale_values: read_d_vec3(reader)?,
                voxel_size: read_d_vec3(reader)?,
                scale_values_inverse: read_d_vec3(reader)?,
                inv_scale_sqr: read_d_vec3(reader)?,
                inv_twice_scale: read_d_vec3(reader)?,
            },
            "UniformScaleTranslateMap" | "ScaleTranslateMap" => Map::ScaleTranslateMap {
                translation: read_d_vec3(reader)?,
                scale_values: read_d_vec3(reader)?,
                voxel_size: read_d_vec3(reader)?,
                scale_values_inverse: read_d_vec3(reader)?,
                inv_scale_sqr: read_d_vec3(reader)?,
                inv_twice_scale: read_d_vec3(reader)?,
            },
            v => panic!("Not supported {}", v),
        })
    }

    fn read_node_header<ValueTy: Pod>(
        reader: &mut R,
        log_2_dim: u32,
        header: &ArchiveHeader,
        gd: &GridDescriptor,
    ) -> Result<NodeHeader, ParseError> {
        let linear_dim = (1 << (3 * log_2_dim)) as usize;

        let mut child_mask = bitvec![u64, Lsb0; 0; linear_dim];
        let mut value_mask = bitvec![u64, Lsb0; 0; linear_dim];
        reader.read_u64_into::<LittleEndian>(child_mask.as_raw_mut_slice())?;
        reader.read_u64_into::<LittleEndian>(value_mask.as_raw_mut_slice())?;

        let linear_dim = if header.file_version < OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION {
            child_mask.count_zeros()
        } else {
            (1 << (3 * log_2_dim)) as usize
        };

        let data = Self::read_compressed::<ValueTy>(
            reader,
            header,
            gd,
            linear_dim,
            value_mask.as_bitslice(),
        )?;
        let data = bytemuck::cast_slice::<ValueTy, u8>(&data).to_vec();

        Ok(NodeHeader {
            child_mask,
            value_mask,
            data,
            log_2_dim,
        })
    }

    fn read_compressed_data<T: Pod>(
        reader: &mut R,
        _archive: &ArchiveHeader,
        gd: &GridDescriptor,
        count: usize,
    ) -> Result<Vec<T>, ParseError> {
        Ok(if gd.compression.contains(Compression::BLOSC) {
            let num_compressed_bytes = reader.read_i64::<LittleEndian>()?;
            let compressed_count = num_compressed_bytes / std::mem::size_of::<T>() as i64;

            trace!("Reading blosc data, {} bytes", num_compressed_bytes);
            if num_compressed_bytes <= 0 {
                let mut data = vec![T::zeroed(); (-compressed_count) as usize];
                reader.read_exact(cast_slice_mut(&mut data))?;
                assert_eq!(-compressed_count as usize, count);
                data
            } else {
                let mut blosc_data = vec![0u8; num_compressed_bytes as usize];
                reader.read_exact(&mut blosc_data)?;
                if count > 0 {
                    let mut nbytes: usize = 0;
                    let mut cbytes: usize = 0;
                    let mut blocksize: usize = 0;
                    unsafe {
                        blosc_cbuffer_sizes(
                            blosc_data.as_ptr().cast(),
                            &mut nbytes,
                            &mut cbytes,
                            &mut blocksize,
                        )
                    };
                    if nbytes == 0 {
                        return Err(ParseError::UnsupportedBloscFormat);
                    }
                    let dest_size = nbytes / std::mem::size_of::<T>();
                    let mut dest: Vec<T> = vec![Zeroable::zeroed(); dest_size];
                    let error = unsafe {
                        blosc_src::blosc_decompress_ctx(
                            blosc_data.as_ptr().cast(),
                            dest.as_mut_ptr().cast(),
                            nbytes,
                            1,
                        )
                    };
                    if error < 1 {
                        return Err(ParseError::InvalidBloscData);
                    }
                    dest
                } else {
                    trace!(
                        "Skipping blosc decompression because of a {}-count read",
                        count
                    );
                    vec![T::zeroed(); 0]
                }
            }
        } else if gd.compression.contains(Compression::ZIP) {
            let num_zipped_bytes = reader.read_i64::<LittleEndian>()?;
            let compressed_count = num_zipped_bytes / std::mem::size_of::<T>() as i64;

            trace!("Reading zipped data, {} bytes", num_zipped_bytes);
            if num_zipped_bytes <= 0 {
                let mut data = vec![T::zeroed(); (-compressed_count) as usize];
                reader.read_exact(cast_slice_mut(&mut data))?;
                data
            } else {
                let mut zipped_data = vec![0u8; num_zipped_bytes as usize];
                reader.read_exact(&mut zipped_data)?;

                let mut zip_reader = flate2::read::ZlibDecoder::new(zipped_data.as_slice());
                let mut data = vec![T::zeroed(); count];
                zip_reader.read_exact(cast_slice_mut(&mut data))?;
                data
            }
        } else {
            trace!("Reading uncompressed data, {} elements", count);

            let mut data = vec![T::zeroed(); count];
            reader.read_exact(cast_slice_mut(&mut data))?;
            data
        })
    }

    fn read_compressed<T: Pod>(
        reader: &mut R,
        archive: &ArchiveHeader,
        gd: &GridDescriptor,
        num_values: usize,
        value_mask: &BitSlice<u64, Lsb0>,
    ) -> Result<Vec<T>, ParseError> {
        let mut meta_data: NodeMetaData = NodeMetaData::NoMaskAndAllVals;
        if archive.file_version >= OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION {
            meta_data = reader.read_u8()?.try_into()?;
        }

        // jb-todo: proper background value support
        let mut inactive_val0 = T::zeroed();
        let mut inactive_val1 = T::zeroed();
        if meta_data == NodeMetaData::NoMaskAndOneInactiveVal
            || meta_data == NodeMetaData::MaskAndOneInactiveVal
            || meta_data == NodeMetaData::MaskAndTwoInactiveVals
        {
            reader.read_exact(bytes_of_mut(&mut inactive_val0))?;

            if meta_data == NodeMetaData::MaskAndTwoInactiveVals {
                reader.read_exact(bytes_of_mut(&mut inactive_val1))?;
            }
        }

        let mut selection_mask = bitvec![u64, Lsb0; 0; num_values];

        if meta_data == NodeMetaData::MaskAndNoInactiveVals
            || meta_data == NodeMetaData::MaskAndOneInactiveVal
            || meta_data == NodeMetaData::MaskAndTwoInactiveVals
        {
            // let selection_mask = reader.read_u32::<LittleEndian>()?;
            reader.read_u64_into::<LittleEndian>(selection_mask.as_raw_mut_slice())?;
        }

        let count = if gd.compression.contains(Compression::ACTIVE_MASK)
            && meta_data != NodeMetaData::NoMaskAndAllVals
            && archive.file_version >= OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION
        {
            value_mask.count_ones()
        } else {
            num_values
        };

        let data = Self::read_compressed_data(reader, archive, gd, count)?;

        Ok(
            if gd.compression.contains(Compression::ACTIVE_MASK) && data.len() != num_values {
                trace!(
                    "Expanding active mask data {} to {}",
                    data.len(),
                    num_values
                );

                let mut expanded = vec![T::zeroed(); num_values];
                let mut read_idx = 0;
                for dest_idx in 0..num_values {
                    expanded[dest_idx] = if value_mask[dest_idx] {
                        let v = data[read_idx];
                        read_idx += 1;
                        v
                    } else if selection_mask[dest_idx] {
                        inactive_val1
                    } else {
                        inactive_val0
                    }
                }
                expanded
            } else {
                data
            },
        )
    }

    fn read_metadata(reader: &mut R) -> Result<Metadata, ParseError> {
        let meta_data_count = reader.read_u32::<LittleEndian>()?;
        let mut meta_data = Metadata::default();

        for _ in 0..meta_data_count {
            let name = Self::read_name(reader)?;
            let data_type = Self::read_name(reader)?;

            let len = reader.read_u32::<LittleEndian>()?;

            meta_data.0.insert(
                name,
                match data_type.as_str() {
                    "string" => MetadataValue::String(read_string(reader, len as usize)?),
                    "bool" => {
                        let val = reader.read_u8()?;
                        MetadataValue::Bool(val != 0)
                    }
                    "int32" => {
                        let val = reader.read_i32::<LittleEndian>()?;
                        MetadataValue::I32(val)
                    }
                    "int64" => {
                        let val = reader.read_i64::<LittleEndian>()?;
                        MetadataValue::I64(val)
                    }
                    "float" => {
                        let val = reader.read_f32::<LittleEndian>()?;
                        MetadataValue::Float(val)
                    }
                    "vec3i" => MetadataValue::Vec3i(read_i_vec3(reader)?),
                    name => {
                        let mut data = vec![0u8; len as usize];
                        reader.read_exact(&mut data)?;

                        warn!("Unknown metadata value {}", name);

                        MetadataValue::Unknown {
                            name: name.to_owned(),
                            data,
                        }
                    }
                },
            );
        }

        trace!("Metadata");
        for (name, value) in meta_data.0.iter() {
            trace!("{}: {:?}", name, value);
        }

        Ok(meta_data)
    }

    fn read_tree_topology<ValueTy: Pod>(
        header: &ArchiveHeader,
        gd: &GridDescriptor,
        reader: &mut R,
    ) -> Result<Tree, ParseError> {
        let buffer_count = reader.read_u32::<LittleEndian>()?;
        assert_eq!(buffer_count, 1, "Multi-buffer trees are not supported");

        let _root_node_background_value = reader.read_u32::<LittleEndian>()?;
        let number_of_tiles = reader.read_u32::<LittleEndian>()?;
        let number_of_root_nodes = reader.read_u32::<LittleEndian>()?;

        let mut root_nodes = vec![];

        for _tile_idx in 0..number_of_tiles {
            let _vec = read_i_vec3(reader)?;
            let _value = reader.read_u32::<LittleEndian>()?;
            let _active = reader.read_u8()?;
        }

        for _root_idx in 0..number_of_root_nodes {
            let origin = read_i_vec3(reader)?;

            let node_5 =
                Self::read_node_header::<ValueTy>(reader, 5 /* 32 * 32 * 32 */, header, gd)?;
            let mut child_5 = HashMap::default();

            let mut root = Node5 {
                child_mask: node_5.child_mask.clone(),
                value_mask: node_5.value_mask.clone(),
                nodes: Default::default(),
                origin,
            };

            for idx in node_5.child_mask.iter_ones() {
                let node_4 = Self::read_node_header::<ValueTy>(
                    reader, 4, /* 16 * 16 * 16 */
                    header, gd,
                )?;
                let mut child_4 = HashMap::default();

                let mut cur_node_4 = Node4 {
                    child_mask: node_4.child_mask.clone(),
                    value_mask: node_4.value_mask.clone(),
                    nodes: Default::default(),
                    origin: root.offset_to_global_coord(Index(idx as u32)).0,
                };

                for idx in node_4.child_mask.iter_ones() {
                    let linear_dim = (1 << (3 * 3)) as usize;

                    let mut value_mask = bitvec![u64, Lsb0; 0; linear_dim];
                    reader.read_u64_into::<LittleEndian>(value_mask.as_raw_mut_slice())?;

                    child_4.insert(
                        idx as u32,
                        Node3 {
                            buffer: vec![],
                            value_mask,
                            origin: cur_node_4.offset_to_global_coord(Index(idx as u32)).0,
                        },
                    );
                }

                cur_node_4.nodes = child_4;

                child_5.insert(idx as u32, cur_node_4);
            }

            root.nodes = child_5;
            root_nodes.push(root);
        }

        Ok(Tree { root_nodes })
    }

    fn read_tree_data<ValueTy: Pod>(
        header: &ArchiveHeader,
        gd: &GridDescriptor,
        reader: &mut R,
        tree: &mut Tree,
    ) -> Result<(), ParseError> {
        gd.seek_to_blocks(reader)?;

        for root_idx in 0..tree.root_nodes.len() {
            let node_5 = &mut tree.root_nodes[root_idx];
            for idx in node_5.child_mask.iter_ones() {
                let node_4 = node_5.nodes.get_mut(&(idx as u32)).unwrap();

                for idx in node_4.child_mask.iter_ones() {
                    let node_3 = node_4.nodes.get_mut(&(idx as u32)).unwrap();

                    let linear_dim = (1 << (3 * 3)) as usize;
                    let mut value_mask = bitvec![u64, Lsb0; 0; linear_dim];
                    reader.read_u64_into::<LittleEndian>(value_mask.as_raw_mut_slice())?;

                    if header.file_version < OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION {
                        node_3.origin = read_i_vec3(reader)?;
                        let num_buffers = reader.read_u8()?;
                        assert_eq!(num_buffers, 1);
                    }

                    let data = Self::read_compressed::<ValueTy>(
                        reader,
                        header,
                        gd,
                        linear_dim,
                        value_mask.as_bitslice(),
                    )?;
                    let data = bytemuck::cast_slice::<ValueTy, u8>(&data).to_vec();

                    node_3.buffer = data;
                }
            }
        }

        Ok(())
    }

    fn read_grid_internal(
        header: &ArchiveHeader,
        reader: &mut R,
        gd: GridDescriptor,
    ) -> Result<Grid, ParseError> {
        gd.seek_to_grid(reader).unwrap();
        // Having to re-do this is ugly, as we already did this while parsing the descriptor
        if header.file_version >= OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION {
            let _: Compression = reader.read_u32::<LittleEndian>()?.try_into().unwrap();
        }
        let _ = Self::read_metadata(reader).unwrap();

        if header.file_version >= OPENVDB_FILE_VERSION_GRID_INSTANCING {
            let transform = Self::read_transform(reader)?;

            let tree = match gd.leaf_data_type() {
                LeafDataType::F32 => {
                    let mut tree = Self::read_tree_topology::<f32>(header, &gd, reader)?;
                    Self::read_tree_data::<f32>(header, &gd, reader, &mut tree)?;
                    tree
                }
                LeafDataType::F16 => {
                    let mut tree = Self::read_tree_topology::<f16>(header, &gd, reader)?;
                    Self::read_tree_data::<f16>(header, &gd, reader, &mut tree)?;
                    tree
                }
                LeafDataType::Float3 => todo!(),
            };

            Ok(Grid {
                tree,
                transform,
                grid_descriptor: gd,
            })
        } else {
            todo!("Old file version not supported {}", header.file_version);
        }
    }

    fn read_grid_descriptors(
        header: &ArchiveHeader,
        reader: &mut R,
    ) -> Result<BTreeMap<String, GridDescriptor>, ParseError> {
        // Should be guaranteed by minimum file version
        assert!(header.has_grid_offsets);

        let mut result = BTreeMap::new();
        for _ in 0..header.grid_count {
            let name = Self::read_name(reader)?;
            let grid_type = Self::read_name(reader)?;

            let instance_parent = if header.file_version >= OPENVDB_FILE_VERSION_GRID_INSTANCING {
                Self::read_name(reader)?
            } else {
                todo!("instance_parent, file version: {}", header.file_version)
            };

            let grid_pos = reader.read_u64::<LittleEndian>()?;
            let block_pos = reader.read_u64::<LittleEndian>()?;
            let end_pos = reader.read_u64::<LittleEndian>()?;

            let mut gd = GridDescriptor {
                name: name.clone(),
                grid_type,
                instance_parent,
                grid_pos,
                block_pos,
                end_pos,
                compression: header.compression,
                meta_data: Default::default(),
            };

            gd.seek_to_grid(reader)?;
            if header.file_version >= OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION {
                gd.compression = reader.read_u32::<LittleEndian>()?.try_into()?;
            }
            gd.meta_data = Self::read_metadata(reader)?;

            assert!(
                result.insert(name.clone(), gd).is_none(),
                "Grid named {name} already exists"
            );

            reader.seek(SeekFrom::Start(end_pos))?;
        }

        Ok(result)
    }
}

impl TryFrom<u8> for NodeMetaData {
    type Error = ParseError;

    fn try_from(v: u8) -> Result<NodeMetaData, ParseError> {
        Ok(match v {
            0 => Self::NoMaskOrInactiveVals,
            1 => Self::NoMaskAndMinusBg,
            2 => Self::NoMaskAndOneInactiveVal,
            3 => Self::MaskAndNoInactiveVals,
            4 => Self::MaskAndOneInactiveVal,
            5 => Self::MaskAndTwoInactiveVals,
            6 => Self::NoMaskAndAllVals,
            _ => return Err(ParseError::InvalidNodeMetadata(v)),
        })
    }
}

impl TryFrom<u32> for Compression {
    type Error = ParseError;

    fn try_from(v: u32) -> Result<Compression, ParseError> {
        Self::from_bits(v).ok_or(ParseError::InvalidCompression(v))
    }
}
