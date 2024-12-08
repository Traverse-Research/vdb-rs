use crate::coordinates::Index;
use crate::data_structure::{
    ArchiveHeader, Compression, Grid, GridDescriptor, Metadata, MetadataValue, Node, Node3, Node4,
    Node5, NodeHeader, NodeMetaData, Tree,
};
use crate::transform::Map;

use bitvec::prelude::*;
use blosc_src::blosc_cbuffer_sizes;
use bytemuck::{bytes_of_mut, cast, cast_slice, cast_slice_mut, Pod, Zeroable};
use byteorder::{LittleEndian, ReadBytesExt};

use half::f16;
use log::{trace, warn};
use std::any::TypeId;
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom};
use std::mem::size_of;

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
    let vec = glam::DVec3::new(x, y, z);
    Ok(vec)
}

fn read_i_vec3<R: Read + Seek>(reader: &mut R) -> Result<glam::IVec3, ParseError> {
    let x = reader.read_i32::<LittleEndian>()?;
    let y = reader.read_i32::<LittleEndian>()?;
    let z = reader.read_i32::<LittleEndian>()?;
    let vec = glam::IVec3::new(x, y, z);
    Ok(vec)
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
        dbg!(&magic);
        if magic == 0x2042445600000000 {
            return Err(ParseError::MagicMismatch);
        }

        let file_version = reader.read_u32::<LittleEndian>()?;
        dbg!(&file_version);
        if file_version < OPENVDB_MIN_SUPPORTED_VERSION {
            return Err(ParseError::UnsupportedVersion(file_version));
        }

        // Stored from version 211 onward, our minimum supported version is 213
        let library_version_major = reader.read_u32::<LittleEndian>()?;
        let library_version_minor = reader.read_u32::<LittleEndian>()?;
        dbg!(&library_version_major);
        dbg!(&library_version_minor);

        // Stored from version 212 onward, our minimum supported version is 213
        let has_grid_offsets = reader.read_u8()? == 1;
        dbg!(&has_grid_offsets);

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
            dbg!(is_compressed);
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
        dbg!(&guid);

        let meta_data = Self::read_metadata(&mut reader)?;
        let grid_count = reader.read_u32::<LittleEndian>()?;
        dbg!(&grid_count);

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

    pub fn read_grid<ExpectedTy: Pod>(
        &mut self,
        name: &str,
    ) -> Result<Grid<ExpectedTy>, ParseError> {
        let grid_descriptor = self.grid_descriptors.get(name).cloned();
        let mut gd = grid_descriptor.ok_or_else(|| ParseError::InvalidGridName(name.to_owned()))?;
        gd.meta_data.0.insert(
            "file_delayed_load".to_string(),
            MetadataValue::Unknown {
                name: "__delayedload".to_string(),
                data: vec![0],
            },
        );
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
        dbg!(&name);

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
    ) -> Result<NodeHeader<ValueTy>, ParseError> {
        dbg!("--------------READING NODE HEADER--------------------");
        let linear_dim = (1 << (3 * log_2_dim)) as usize;

        let mut child_mask = bitvec![u64, Lsb0; 0; linear_dim];
        let mut value_mask = bitvec![u64, Lsb0; 0; linear_dim];
        reader.read_u64_into::<LittleEndian>(child_mask.as_raw_mut_slice())?;
        reader.read_u64_into::<LittleEndian>(value_mask.as_raw_mut_slice())?;
        dbg!(child_mask.count_ones());
        dbg!(child_mask.count_zeros());
        dbg!(value_mask.count_ones());
        dbg!(value_mask.count_zeros());

        let linear_dim = if header.file_version < OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION {
            child_mask.count_zeros()
        } else {
            (1 << (3 * log_2_dim)) as usize
        };
        dbg!(linear_dim);

        let data = Self::read_compressed(reader, header, gd, linear_dim, value_mask.as_bitslice())?;

        Ok(NodeHeader {
            child_mask,
            value_mask,
            data,
            log_2_dim,
        })
    }

    fn print_reader_scope(reader: &mut R) {
        let length = 256;
        reader
            .seek(SeekFrom::Current(-length * size_of::<i64>() as i64))
            .unwrap();
        for i in 0..(length * 2) {
            let int1 = reader.read_i16::<LittleEndian>().unwrap();
            let int2 = reader.read_i16::<LittleEndian>().unwrap();
            let int3 = reader.read_i16::<LittleEndian>().unwrap();
            let int4 = reader.read_i16::<LittleEndian>().unwrap();
            print!(
                "{i:05} {:010} {:018b} {:018b} {:018b} {:018b} {:010} {:010} {:010} {:010}",
                reader.stream_position().unwrap(),
                int1,
                int2,
                int3,
                int4,
                int1,
                int2,
                int3,
                int4
            );
            if i == length {
                print!(" <<<");
            }
            println!("");
        }
        reader
            .seek(SeekFrom::Current(-length * size_of::<i64>() as i64))
            .unwrap();
    }

    fn read_compressed_data<T: Pod>(
        reader: &mut R,
        _archive: &ArchiveHeader,
        gd: &GridDescriptor,
        count: usize,
    ) -> Result<Vec<T>, ParseError> {
        Ok(if gd.compression.contains(Compression::BLOSC) {
            if count <= 0 {
                return Ok(vec![Zeroable::zeroed(); count]);
            }
            dbg!(&count);
            //Self::print_reader_scope(reader);
            let mut num_compressed_bytes_slice = [0i32, 0i32];
            reader.read_exact(cast_slice_mut(num_compressed_bytes_slice.as_mut_slice()))?;
            let num_compressed_bytes = num_compressed_bytes_slice[0] as i64;
            //let num_compressed_bytes = reader.read_i64::<LittleEndian>()?;
            dbg!(&num_compressed_bytes);
            let compressed_count = num_compressed_bytes / std::mem::size_of::<T>() as i64;

            trace!("Reading blosc data, {} bytes", num_compressed_bytes);
            if num_compressed_bytes <= 0 {
                assert_eq!(-compressed_count as usize, count);
                let mut data = vec![T::zeroed(); (-compressed_count) as usize];
                reader.read_exact(cast_slice_mut(&mut data))?;
                dbg!(data.len());
                data
            } else {
                let mut blosc_data = vec![0u8; num_compressed_bytes as usize];
                reader.read_exact(&mut blosc_data)?;
                dbg!(blosc_data.len());
                //if count > 0 {
                //let mut nbytes: usize = 0;
                //let mut cbytes: usize = 0;
                //let mut blocksize: usize = 0;
                //unsafe {
                //    blosc_cbuffer_sizes(
                //        blosc_data.as_ptr().cast(),
                //        &mut nbytes,
                //        &mut cbytes,
                //        &mut blocksize,
                //    )
                //};
                //if nbytes == 0 {
                //    return Err(ParseError::UnsupportedBloscFormat);
                //}
                //let dest_size = nbytes / std::mem::size_of::<T>();
                let mut dest: Vec<T> = vec![Zeroable::zeroed(); count];
                let error = unsafe {
                    blosc_src::blosc_decompress_ctx(
                        blosc_data.as_ptr().cast(),
                        dest.as_mut_ptr().cast(),
                        count * size_of::<T>(),
                        1,
                    )
                };
                dbg!(&blosc_data);
                dbg!(&cast_slice::<T, f16>(&dest));
                if error < 1 {
                    return Err(ParseError::InvalidBloscData);
                }
                if error != count as i32 * std::mem::size_of::<T>() as i32 {
                    panic!(
                        "expected this to be equal but got {} and {}",
                        error,
                        count * std::mem::size_of::<T>()
                    );
                }
                dest
                //} else {
                //    trace!(
                //        "Skipping blosc decompression because of a {}-count read",
                //        count
                //    );
                //    vec![T::zeroed(); 0]
                //}
            }
        } else if gd.compression.contains(Compression::ZIP) {
            let num_zipped_bytes = reader.read_i64::<LittleEndian>()?;
            dbg!(&num_zipped_bytes);
            let compressed_count = num_zipped_bytes / std::mem::size_of::<T>() as i64;

            trace!("Reading zipped data, {} bytes", num_zipped_bytes);
            if num_zipped_bytes <= 0 {
                let mut data = vec![T::zeroed(); (-compressed_count) as usize];
                reader.read_exact(cast_slice_mut(&mut data))?;
                dbg!(&data.len());
                data
            } else {
                let mut zipped_data = vec![0u8; num_zipped_bytes as usize];
                reader.read_exact(&mut zipped_data)?;
                dbg!(&zipped_data.len());

                let mut zip_reader = flate2::read::ZlibDecoder::new(zipped_data.as_slice());
                let mut data = vec![T::zeroed(); count];
                zip_reader.read_exact(cast_slice_mut(&mut data))?;
                data
            }
        } else {
            trace!("Reading uncompressed data, {} elements", count);

            let mut data = vec![T::zeroed(); count];
            reader.read_exact(cast_slice_mut(&mut data))?;
            dbg!(&data.len());
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
        dbg!("--------------------READING COMPRESSION----------------------");
        if archive.file_version >= OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION {
            meta_data = reader.read_u8()?.try_into()?;
            dbg!(meta_data);
        }

        // jb-todo: proper background value support
        let mut inactive_val0 = T::zeroed();
        let mut inactive_val1 = T::zeroed();
        if meta_data == NodeMetaData::NoMaskAndOneInactiveVal
            || meta_data == NodeMetaData::MaskAndOneInactiveVal
            || meta_data == NodeMetaData::MaskAndTwoInactiveVals
        {
            reader.read_exact(bytes_of_mut(&mut inactive_val0))?;
            dbg!(&cast::<T, f16>(inactive_val0));

            if meta_data == NodeMetaData::MaskAndTwoInactiveVals {
                reader.read_exact(bytes_of_mut(&mut inactive_val1))?;
                dbg!(&cast::<T, f16>(inactive_val1));
            }
        }

        let mut selection_mask = bitvec![u64, Lsb0; 0; num_values];

        if meta_data == NodeMetaData::MaskAndNoInactiveVals
            || meta_data == NodeMetaData::MaskAndOneInactiveVal
            || meta_data == NodeMetaData::MaskAndTwoInactiveVals
        {
            // let selection_mask = reader.read_u32::<LittleEndian>()?;
            reader.read_u64_into::<LittleEndian>(selection_mask.as_raw_mut_slice())?;
            dbg!(selection_mask.count_ones());
            dbg!(selection_mask.count_zeros());
        }

        let count = if gd.compression.contains(Compression::ACTIVE_MASK)
            && meta_data != NodeMetaData::NoMaskAndAllVals
            && archive.file_version >= OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION
        {
            value_mask.count_ones()
        } else {
            num_values
        };

        // jb-todo: we may need to extend this to vector types
        let data = if gd.meta_data.is_half_float()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            dbg!("WTF YOU ARE SUPPOSED TO BE A F32");
            let data = Self::read_compressed_data::<f16>(reader, archive, gd, count)?;
            bytemuck::cast_vec(data.into_iter().map(f16::to_f32).collect::<Vec<f32>>())
        } else if !gd.meta_data.is_half_float()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f16>()
        {
            dbg!("WTF YOU ARE SUPPOSED TO BE A F16");
            let data = Self::read_compressed_data::<f32>(reader, archive, gd, count)?;
            bytemuck::cast_vec(data.into_iter().map(f16::from_f32).collect::<Vec<_>>())
        } else {
            dbg!("ALL AS EXPECTED");
            Self::read_compressed_data::<T>(reader, archive, gd, count)?
        };

        Ok(
            if gd.compression.contains(Compression::ACTIVE_MASK) && data.len() != num_values {
                dbg!(
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
        dbg!(&meta_data_count);
        let mut meta_data = Metadata::default();

        for _ in 0..meta_data_count {
            let name = Self::read_name(reader)?;
            dbg!(&name);
            let data_type = Self::read_name(reader)?;
            dbg!(&data_type);

            let len = reader.read_u32::<LittleEndian>()?;
            dbg!(&len);

            meta_data.0.insert(
                name,
                match data_type.as_str() {
                    "string" => MetadataValue::String(read_string(reader, len as usize)?),
                    "bool" => {
                        let val = reader.read_u8()?;
                        dbg!(&val);
                        MetadataValue::Bool(val != 0)
                    }
                    "int32" => {
                        let val = reader.read_i32::<LittleEndian>()?;
                        dbg!(&val);
                        MetadataValue::I32(val)
                    }
                    "int64" => {
                        let val = reader.read_i64::<LittleEndian>()?;
                        dbg!(&val);
                        MetadataValue::I64(val)
                    }
                    "float" => {
                        let val = reader.read_f32::<LittleEndian>()?;
                        dbg!(&val);
                        MetadataValue::Float(val)
                    }
                    "vec3i" => {
                        let val = read_i_vec3(reader)?;
                        dbg!(&val);
                        MetadataValue::Vec3i(val)
                    }
                    name => {
                        let mut data = vec![0u8; len as usize];
                        reader.read_exact(&mut data)?;
                        dbg!(&data.len());

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
    ) -> Result<Tree<ValueTy>, ParseError> {
        let buffer_count = reader.read_u32::<LittleEndian>()?;
        dbg!(buffer_count);
        assert_eq!(buffer_count, 1, "Multi-buffer trees are not supported");

        let _root_node_background_value = reader.read_u32::<LittleEndian>()?;
        dbg!(_root_node_background_value);
        let number_of_tiles = reader.read_u32::<LittleEndian>()?;
        dbg!(number_of_tiles);
        let number_of_root_nodes = reader.read_u32::<LittleEndian>()?;
        dbg!(number_of_root_nodes);

        let mut root_nodes = vec![];

        for _tile_idx in 0..number_of_tiles {
            let _vec = read_i_vec3(reader)?;
            dbg!(_vec);
            let _value = reader.read_u32::<LittleEndian>()?;
            dbg!(_value);
            let _active = reader.read_u8()?;
            dbg!(_active);
        }

        dbg!(&header);
        dbg!(&gd);

        for _root_idx in 0..number_of_root_nodes {
            dbg!("----------------READING ROOT NODE TOPOLOGY------------");
            let origin = read_i_vec3(reader)?;
            dbg!(origin);

            let node_5 =
                Self::read_node_header::<ValueTy>(reader, 5 /* 32 * 32 * 32 */, header, gd)?;
            let mut child_5 = BTreeMap::default();

            let mut root = Node5 {
                child_mask: node_5.child_mask.clone(),
                value_mask: node_5.value_mask.clone(),
                nodes: Default::default(),
                data: node_5.data,
                origin,
            };

            for idx in node_5.child_mask.iter_ones() {
                dbg!("----------------READING 4 NODE TOPOLOGY------------");
                let node_4 = Self::read_node_header::<ValueTy>(
                    reader, 4, /* 16 * 16 * 16 */
                    header, gd,
                )?;
                let mut child_4 = BTreeMap::default();

                let mut cur_node_4 = Node4 {
                    child_mask: node_4.child_mask.clone(),
                    value_mask: node_4.value_mask.clone(),
                    nodes: Default::default(),
                    data: node_4.data,
                    origin: root.offset_to_global_coord(Index(idx as u32)).0,
                };

                for idx in node_4.child_mask.iter_ones() {
                    dbg!("----------------READING 3 NODE TOPOLOGY------------");
                    let linear_dim = (1 << (3 * 3)) as usize;

                    let mut value_mask = bitvec![u64, Lsb0; 0; linear_dim];
                    reader.read_u64_into::<LittleEndian>(value_mask.as_raw_mut_slice())?;
                    dbg!(value_mask.count_ones());
                    dbg!(value_mask.count_zeros());

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
        tree: &mut Tree<ValueTy>,
    ) -> Result<(), ParseError> {
        gd.seek_to_blocks(reader)?;
        println!("----------------------------------------------");

        for root_idx in 0..tree.root_nodes.len() {
            let node_5 = &mut tree.root_nodes[root_idx];
            for idx in node_5.child_mask.iter_ones() {
                let node_4 = node_5.nodes.get_mut(&(idx as u32)).unwrap();

                for idx in node_4.child_mask.iter_ones() {
                    dbg!("--------------READING TREE DATA--------------------");
                    let node_3 = node_4.nodes.get_mut(&(idx as u32)).unwrap();

                    let linear_dim = (1 << (3 * 3)) as usize;
                    let mut value_mask = bitvec![u64, Lsb0; 0; linear_dim];
                    reader.read_u64_into::<LittleEndian>(value_mask.as_raw_mut_slice())?;
                    dbg!(value_mask.count_ones());
                    dbg!(value_mask.count_zeros());

                    if header.file_version < OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION {
                        node_3.origin = read_i_vec3(reader)?;
                        dbg!(node_3.origin);
                        let num_buffers = reader.read_u8()?;
                        dbg!(num_buffers);
                        assert_eq!(num_buffers, 1);
                    }

                    let data = Self::read_compressed(
                        reader,
                        header,
                        gd,
                        linear_dim,
                        value_mask.as_bitslice(),
                    )?;

                    node_3.buffer = data;
                }
            }
        }

        Ok(())
    }

    fn read_grid_internal<ValueTy: Pod>(
        header: &ArchiveHeader,
        reader: &mut R,
        gd: GridDescriptor,
    ) -> Result<Grid<ValueTy>, ParseError> {
        gd.seek_to_grid(reader).unwrap();
        // Having to re-do this is ugly, as we already did this while parsing the descriptor
        if header.file_version >= OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION {
            let _: Compression = reader.read_u32::<LittleEndian>()?.try_into().unwrap();
        }
        let _ = Self::read_metadata(reader).unwrap();

        if header.file_version >= OPENVDB_FILE_VERSION_GRID_INSTANCING {
            let transform = Self::read_transform(reader)?;
            dbg!(&transform);
            let mut tree = Self::read_tree_topology(header, &gd, reader)?;
            Self::read_tree_data(header, &gd, reader, &mut tree)?;

            Ok(Grid {
                tree,
                transform,
                descriptor: gd,
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
            dbg!(&name);
            let grid_type = Self::read_name(reader)?;
            dbg!(&grid_type);

            let instance_parent = if header.file_version >= OPENVDB_FILE_VERSION_GRID_INSTANCING {
                Self::read_name(reader)?
            } else {
                todo!("instance_parent, file version: {}", header.file_version)
            };
            dbg!(&instance_parent);

            let grid_pos = reader.read_u64::<LittleEndian>()?;
            dbg!(grid_pos);
            let block_pos = reader.read_u64::<LittleEndian>()?;
            dbg!(block_pos);
            let end_pos = reader.read_u64::<LittleEndian>()?;
            dbg!(end_pos);

            let mut gd = GridDescriptor {
                name: name.clone(),
                file_version: header.file_version,
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
                dbg!(gd.compression);
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
