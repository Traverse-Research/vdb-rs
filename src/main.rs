use bitflags::bitflags;
use bitvec::prelude::*;
use bytemuck::{bytes_of_mut, cast_slice_mut, Pod};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use half::f16;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::io::{Read, Seek, SeekFrom};

const OPENVDB_FILE_VERSION_ROOTNODE_MAP: u32 = 213;
const OPENVDB_FILE_VERSION_INTERNALNODE_COMPRESSION: u32 = 214;
const OPENVDB_FILE_VERSION_SIMPLIFIED_GRID_TYPENAME: u32 = 215;
const OPENVDB_FILE_VERSION_GRID_INSTANCING: u32 = 216;
const OPENVDB_FILE_VERSION_BOOL_LEAF_OPTIMIZATION: u32 = 217;
const OPENVDB_FILE_VERSION_BOOST_UUID: u32 = 218;
const OPENVDB_FILE_VERSION_NO_GRIDMAP: u32 = 219;
const OPENVDB_FILE_VERSION_NEW_TRANSFORM: u32 = 219;
const OPENVDB_FILE_VERSION_SELECTIVE_COMPRESSION: u32 = 220;
const OPENVDB_FILE_VERSION_FLOAT_FRUSTUM_BBOX: u32 = 221;
const OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION: u32 = 222;
const OPENVDB_FILE_VERSION_BLOSC_COMPRESSION: u32 = 223;
const OPENVDB_FILE_VERSION_POINT_INDEX_GRID: u32 = 223;
const OPENVDB_FILE_VERSION_MULTIPASS_IO: u32 = 22;

#[derive(thiserror::Error, Debug)]
enum ParseError {
    #[error("Magic bytes mismatched")]
    MagicMismatch,
    #[error("Invalid compression")]
    InvalidCompression,
    #[error("Invalid node meta-data")]
    InvalidNodeMetadata,
    #[error("Unsupported compression")]
    UnsupportedCompression,
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

fn read_name<R: Read + Seek>(reader: &mut R) -> Result<String, ParseError> {
    let len = reader.read_u32::<LittleEndian>()? as usize;
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
struct Grid<ValueTy> {
    tree: Tree<ValueTy>,
    transform: Map,
    grid_descriptor: GridDescriptor,
}

#[derive(Debug)]
struct GridDescriptor {
    name: String,
    grid_type: String,
    instance_parent: String,
    grid_pos: u64,
    block_pos: u64,
    end_pos: u64,
    compression: Compression,
    meta_data: Metadata,
}

impl GridDescriptor {
    fn seek_to_grid<R: Read + Seek>(&self, reader: &mut R) {
        reader.seek(SeekFrom::Start(self.grid_pos));
    }

    fn seek_to_blocks<R: Read + Seek>(&self, reader: &mut R) {
        reader.seek(SeekFrom::Start(self.block_pos));
    }
}

fn read_grid<R: Read + Seek, ValueTy: Pod + std::fmt::Debug>(
    header: ArchiveHeader,
    reader: &mut R,
) -> Result<Grid<ValueTy>, ParseError> {
    let name = read_name(reader)?;
    let grid_type = read_name(reader)?;

    let instance_parent = if header.file_version >= OPENVDB_FILE_VERSION_GRID_INSTANCING {
        read_name(reader)?
    } else {
        todo!("instance_parent, file version: {}", header.file_version)
    };

    let grid_pos = reader.read_u64::<LittleEndian>()?;
    let block_pos = reader.read_u64::<LittleEndian>()?;
    let end_pos = reader.read_u64::<LittleEndian>()?;

    let mut gd = GridDescriptor {
        name,
        grid_type,
        instance_parent,
        grid_pos,
        block_pos,
        end_pos,
        compression: header.compression,
        meta_data: Default::default(),
    };

    dbg!(&gd);

    gd.seek_to_grid(reader);
    if header.file_version >= OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION {
        gd.compression = reader.read_u32::<LittleEndian>()?.try_into()?;
    }
    gd.meta_data = dbg!(read_metadata(reader)?);

    if header.file_version >= OPENVDB_FILE_VERSION_GRID_INSTANCING {
        let transform = read_transform(reader)?;
        let mut tree = read_tree_topology(&header, &gd, reader)?;
        read_tree_data(&header, &gd, reader, &mut tree)?;

        Ok(Grid {
            tree,
            transform,
            grid_descriptor: gd,
        })
    } else {
        todo!("Old file version not supported {}", header.file_version);
    }
}

#[derive(Debug, Default)]
struct Metadata(HashMap<String, MetadataValue>);

impl Metadata {
    fn is_half_float(&self) -> bool {
        self.0.get("is_saved_as_half_float") == Some(&MetadataValue::Bool(true))
    }
}

#[derive(Debug, Eq, PartialEq)]
enum MetadataValue {
    String(String),
    Vec3i(glam::IVec3),
    I64(i64),
    Bool(bool),
}

fn read_metadata<R: Read + Seek>(reader: &mut R) -> Result<Metadata, ParseError> {
    let meta_data_count = reader.read_u32::<LittleEndian>()?;
    let mut meta_data = Metadata::default();

    for _ in 0..meta_data_count {
        let name = read_name(reader)?;
        let data_type = read_name(reader)?;

        meta_data.0.insert(
            name,
            match data_type.as_str() {
                "string" => MetadataValue::String(read_name(reader)?),
                "bool" => {
                    let len = reader.read_i32::<LittleEndian>()?;
                    let val = reader.read_u8()?;
                    MetadataValue::Bool(val != 0)
                }
                "int64" => {
                    let len = reader.read_i32::<LittleEndian>()?;
                    let val = reader.read_i64::<LittleEndian>()?;
                    MetadataValue::I64(val)
                }
                "vec3i" => {
                    let len = reader.read_i32::<LittleEndian>()?;
                    MetadataValue::Vec3i(read_i_vec3(reader)?)
                }
                v => panic!("Invalid datatype {:?}", v),
            },
        );
    }

    Ok(meta_data)
}

#[derive(Debug)]
enum Map {
    UniformScaleMap {
        scale_values: glam::DVec3,
        voxel_size: glam::DVec3,
        scale_values_inverse: glam::DVec3,
        inv_scale_sqr: glam::DVec3,
        inv_twice_scale: glam::DVec3,
    },
}

fn read_transform<R: Read + Seek>(reader: &mut R) -> Result<Map, ParseError> {
    let name = read_name(reader)?;

    Ok(match name.as_str() {
        "UniformScaleMap" => Map::UniformScaleMap {
            scale_values: read_d_vec3(reader)?,
            voxel_size: read_d_vec3(reader)?,
            scale_values_inverse: read_d_vec3(reader)?,
            inv_scale_sqr: read_d_vec3(reader)?,
            inv_twice_scale: read_d_vec3(reader)?,
        },
        v => panic!("Not supported {}", v),
    })
}

struct GlobalCoord(glam::IVec3);
struct LocalCoord(glam::UVec3);
struct Index(u32);

trait Node {
    const DIM: u32 = 1 << Self::LOG_2_DIM;
    const LOG_2_DIM: u32;
    const TOTAL: u32;

    fn local_coord_to_offset(&self, xyz: LocalCoord) -> Index {
        Index(
            (((xyz.0[0] & (Self::DIM - 1)) >> Self::TOTAL) << 2 * Self::LOG_2_DIM)
                + (((xyz.0[1] & (Self::DIM - 1)) >> Self::TOTAL) << Self::LOG_2_DIM)
                + ((xyz.0[2] & (Self::DIM - 1)) >> Self::TOTAL),
        )
    }

    fn offset_to_local_coord(&self, offset: Index) -> LocalCoord {
        assert!(
            offset.0 < (1 << 3 * Self::LOG_2_DIM),
            "Offset {} out of bounds",
            offset.0
        );

        let x = offset.0 >> 2 * Self::LOG_2_DIM;
        let offset = offset.0 & ((1 << 2 * Self::LOG_2_DIM) - 1);

        let y = offset >> Self::LOG_2_DIM;
        let z = offset & ((1 << Self::LOG_2_DIM) - 1);

        LocalCoord(glam::UVec3::new(x, y, z))
    }

    fn offset_to_global_coord(&self, offset: Index) -> GlobalCoord {
        let mut local_coord = self.offset_to_local_coord(offset);
        local_coord.0[0] <<= Self::TOTAL; //??? Maybe incorrect
        local_coord.0[1] <<= Self::TOTAL; //??? Maybe incorrect
        local_coord.0[2] <<= Self::TOTAL; //??? Maybe incorrect
        GlobalCoord(local_coord.0.as_ivec3() + self.offset())
    }

    fn offset(&self) -> glam::IVec3 {
        glam::IVec3::ZERO
    }
}

#[derive(Debug)]
struct NodeHeader<ValueTy> {
    child_mask: BitVec<u64, Lsb0>,
    value_mask: BitVec<u64, Lsb0>,
    data: Vec<ValueTy>,
    log_2_dim: u32,
}

#[derive(Debug)]
struct Node3<ValueTy> {
    buffer: Vec<ValueTy>,
    value_mask: BitVec<u64, Lsb0>,
    origin: glam::IVec3,
}

impl<ValueTy> Node for Node3<ValueTy> {
    const LOG_2_DIM: u32 = 3;
    const TOTAL: u32 = 0;
}

#[derive(Debug)]
struct Node4<ValueTy> {
    child_mask: BitVec<u64, Lsb0>,
    value_mask: BitVec<u64, Lsb0>,
    nodes: HashMap<u32, Node3<ValueTy>>,
}

impl<ValueTy> Node for Node4<ValueTy> {
    const LOG_2_DIM: u32 = 4;
    const TOTAL: u32 = 3;
}

#[derive(Debug)]
struct Node5<ValueTy> {
    child_mask: BitVec<u64, Lsb0>,
    value_mask: BitVec<u64, Lsb0>,
    nodes: HashMap<u32, Node4<ValueTy>>,
}

impl<ValueTy> Node for Node5<ValueTy> {
    const LOG_2_DIM: u32 = 5;
    const TOTAL: u32 = 7;
}

/*
Interior node

    UnionType mNodes[NUM_VALUES];
    NodeMaskType mChildMask, mValueMask;
    /// Global grid index coordinates (x,y,z) of the local origin of this node
    Coord mOrigin;
*/

/*
Leaf
    /// Buffer containing the actual data values
    Buffer mBuffer;
    /// Bitmask that determines which voxels are active
    NodeMaskType mValueMask;
    /// Global grid index coordinates (x,y,z) of the local origin of this node
    Coord mOrigin;
*/

fn read_compressed<R: Read + Seek, T: Pod>(
    reader: &mut R,
    log_2_dim: u32,
    archive: &ArchiveHeader,
    gd: &GridDescriptor,
    num_values: usize,
    value_mask: &BitSlice<u64, Lsb0>,
) -> Result<Vec<T>, ParseError> {
    let mut meta_data: NodeMetaData = NodeMetaData::NoMaskOrInactiveVals;
    if archive.file_version >= OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION {
        meta_data = reader.read_u8()?.try_into()?;

        if meta_data == NodeMetaData::NoMaskAndOneInactiveVal
            || meta_data == NodeMetaData::MaskAndOneInactiveVal
            || meta_data == NodeMetaData::MaskAndTwoInactiveVals
        {
            todo!("Test this branch");
            let inactive_val0 = reader.read_u32::<LittleEndian>()?;

            if meta_data == NodeMetaData::MaskAndTwoInactiveVals {
                let inactive_val1 = reader.read_u32::<LittleEndian>()?;
            }
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

    let data = if gd.compression.contains(Compression::ZIP) {
        let num_zipped_bytes = reader.read_i64::<LittleEndian>()?;
        if num_zipped_bytes < 0 {
            let mut data = vec![T::zeroed(); (-num_zipped_bytes) as usize];
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
        let mut data = vec![T::zeroed(); count];

        reader.read_exact(cast_slice_mut(&mut data))?;
        data
    };

    Ok(data)
}

fn read_node_header<R: Read + Seek, ValueTy: Pod>(
    reader: &mut R,
    log_2_dim: u32,
    header: &ArchiveHeader,
    gd: &GridDescriptor,
) -> Result<NodeHeader<ValueTy>, ParseError> {
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

    let data = read_compressed(
        reader,
        log_2_dim,
        header,
        gd,
        linear_dim,
        value_mask.as_slice(),
    )?;

    Ok(NodeHeader {
        child_mask,
        value_mask,
        data,
        log_2_dim,
    })
}

#[derive(Debug)]
struct Tree<ValueTy> {
    // origin: glam::IVec3
    root_nodes: Vec<Node5<ValueTy>>,
}

fn read_tree_topology<R: Read + Seek, ValueTy: Pod + std::fmt::Debug>(
    header: &ArchiveHeader,
    gd: &GridDescriptor,
    reader: &mut R,
) -> Result<Tree<ValueTy>, ParseError> {
    let buffer_count = reader.read_u32::<LittleEndian>()?;
    assert_eq!(buffer_count, 1, "Multi-buffer trees are not supported");

    let root_node_background_value = reader.read_u32::<LittleEndian>()?;
    let number_of_tiles = reader.read_u32::<LittleEndian>()?;
    let number_of_root_nodes = reader.read_u32::<LittleEndian>()?;

    let mut root_nodes = vec![];

    for n in 0..number_of_tiles {
        let vec = read_i_vec3(reader)?;
        let value = reader.read_u32::<LittleEndian>()?;
        let active = reader.read_u8()?;
    }

    for root_idx in 0..number_of_root_nodes {
        let origin = read_i_vec3(reader)?;

        let node_5 = read_node_header::<_, ValueTy>(reader, 5 /* 32 * 32 * 32 */, header, gd)?;
        let mut child_5 = HashMap::default();

        for idx in node_5.child_mask.iter_ones() {
            let node_4 =
                read_node_header::<_, ValueTy>(reader, 4 /* 16 * 16 * 16 */, header, gd)?;
            let mut child_4 = HashMap::default();

            for idx in node_4.child_mask.iter_ones() {
                let linear_dim = (1 << (3 * 3)) as usize;

                let mut value_mask = bitvec![u64, Lsb0; 0; linear_dim];
                reader.read_u64_into::<LittleEndian>(value_mask.as_raw_mut_slice())?;

                child_4.insert(
                    idx as u32,
                    Node3 {
                        buffer: vec![],
                        value_mask,
                        origin: Default::default(),
                    },
                );
            }

            child_5.insert(
                idx as u32,
                Node4 {
                    child_mask: node_4.child_mask,
                    value_mask: node_4.value_mask,
                    nodes: child_4,
                },
            );
        }

        root_nodes.push(Node5 {
            child_mask: node_5.child_mask,
            value_mask: node_5.value_mask,
            nodes: child_5,
        });
    }

    Ok(Tree { root_nodes })
}

fn read_tree_data<R: Read + Seek, ValueTy: Pod + std::fmt::Debug>(
    header: &ArchiveHeader,
    gd: &GridDescriptor,
    reader: &mut R,
    tree: &mut Tree<ValueTy>,
) -> Result<(), ParseError> {
    gd.seek_to_blocks(reader);

    for root_idx in 0..tree.root_nodes.len() {
        let node_5 = &mut tree.root_nodes[root_idx];
        for idx in node_5.child_mask.iter_ones() {
            let mut node_4 = node_5.nodes.get_mut(&(idx as u32)).unwrap();

            for idx in node_4.child_mask.iter_ones() {
                let mut node_3 = node_4.nodes.get_mut(&(idx as u32)).unwrap();

                let linear_dim = (1 << (3 * 3)) as usize;
                let mut value_mask = bitvec![u64, Lsb0; 0; linear_dim];
                reader.read_u64_into::<LittleEndian>(value_mask.as_raw_mut_slice())?;

                if header.file_version < OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION {
                    node_3.origin = read_i_vec3(reader)?;
                    let num_buffers = reader.read_u8()?;
                    assert_eq!(num_buffers, 1);
                }

                let data =
                    read_compressed(reader, 3, header, gd, linear_dim, value_mask.as_slice())?;

                node_3.buffer = data;
            }
        }
    }

    println!("{:?}", reader.stream_position());

    Ok(())
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum NodeMetaData {
    NoMaskOrInactiveVals,
    NoMaskAndMinusBg,
    NoMaskAndOneInactiveVal,
    MaskAndNoInactiveVals,
    MaskAndOneInactiveVal,
    MaskAndTwoInactiveVals,
    NoMaskAndAllVals,
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
            _ => return Err(ParseError::InvalidNodeMetadata),
        })
    }
}

bitflags! {
    struct Compression: u32 {
        const NONE = 0;
        const ZIP = 0x1;
        const ACTIVE_MASK = 0x2;
        const BLOSC = 0x4;
        const DEFAULT_COMPRESSION = Self::BLOSC.bits | Self::ACTIVE_MASK.bits;
    }
}

impl TryFrom<u32> for Compression {
    type Error = ParseError;

    fn try_from(v: u32) -> Result<Compression, ParseError> {
        Ok(match v {
            0 => Compression::NONE,
            0x1 => Compression::ZIP,
            0x2 => Compression::ACTIVE_MASK,
            0x4 => Compression::BLOSC,
            _ => return Err(ParseError::InvalidCompression),
        })
    }
}

#[derive(Debug)]
struct ArchiveHeader {
    file_version: u32,
    library_version_major: u32,
    library_version_minor: u32,
    has_grid_offsets: bool,
    compression: Compression,
    guid: String,
    meta_data: Metadata,
    grid_count: u32,
}

fn read_vdb<R: Read + Seek, ValueTy: Pod + std::fmt::Debug>(
    reader: &mut R,
) -> Result<Grid<ValueTy>, ParseError> {
    let magic = reader.read_u64::<LittleEndian>()?;
    if magic == 0x2042445600000000 {
        return Err(ParseError::MagicMismatch);
    }

    let file_version = reader.read_u32::<LittleEndian>()?;
    let library_version_major = reader.read_u32::<LittleEndian>()?;
    let library_version_minor = reader.read_u32::<LittleEndian>()?;
    let has_grid_offsets = reader.read_u8()? == 1;

    let compression = if file_version >= OPENVDB_FILE_VERSION_SELECTIVE_COMPRESSION
        && file_version < OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION
    {
        (reader.read_u8()? as u32).try_into()?
    } else {
        Compression::DEFAULT_COMPRESSION
    };

    let guid = if file_version >= OPENVDB_FILE_VERSION_BOOST_UUID {
        read_string(reader, 36)?
    } else {
        todo!("File version {}", file_version);
    };

    let meta_data = read_metadata(reader)?;
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

    read_grid::<_, ValueTy>(dbg!(header), reader)
}

fn main() -> Result<(), Box<dyn Error>> {
    //let f = File::open("C:/Users/Jasper/Downloads/buddha.vdb-1.0.0/buddha.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/bunny.vdb-1.0.0/bunny.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/bunny_cloud.vdb-1.0.0/bunny_cloud.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/cube.vdb-1.0.0/cube.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/crawler.vdb-1.0.0/crawler.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/dragon.vdb-1.0.0/dragon.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/emu.vdb-1.0.0/emu.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/armadillo.vdb-1.0.0/armadillo.vdb")?;

    let f = File::open("C:/Users/Jasper/Downloads/cube.vdb-1.0.0/cube.vdb")?;
    let mut reader = BufReader::new(f);

    let grid = read_vdb::<_, f16>(&mut reader)?;
    // let mut child_mask = bitvec![u64, Lsb0; 0; 2];
    // child_mask.set(0, true);
    // child_mask.set(1, true);
    // child_mask.set(3, true); // oob

    Ok(())
}
