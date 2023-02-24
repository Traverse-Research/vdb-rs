use bitvec::prelude::*;
use byteorder::{BigEndian, LittleEndian, ReadBytesExt, WriteBytesExt};
use half::f16;
use half::vec::HalfBitsVecExt;
use std::collections::HashMap;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::BufReader;
use std::io::{Read, Seek, SeekFrom};

#[derive(thiserror::Error, Debug)]
enum ParseError {
    #[error("Magic bytes mismatched")]
    MagicMismatch,
    #[error("Invalid cast")]
    InvalidCast,
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
    let len = dbg!(reader.read_u32::<LittleEndian>()? as usize);
    let mut string = String::with_capacity(len);
    for _ in 0..len {
        let c = reader.read_u8()? as char;
        string.push(c);
    }
    Ok(dbg!(string))
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

struct Grid {}

fn read_grid<R: Read + Seek>(reader: &mut R) -> Result<Grid, ParseError> {
    let name = read_name(reader)?;
    dbg!(name);

    let grid_type = read_name(reader)?;
    dbg!(grid_type);

    let instance_parent = reader.read_u32::<LittleEndian>()?;
    dbg!(instance_parent);

    // comment says "Grid descriptor stream position"
    let grid_pos = reader.read_u64::<LittleEndian>()?;
    dbg!(grid_pos);

    let block_pos = reader.read_u64::<LittleEndian>()?;
    dbg!(block_pos);

    let end_pos = reader.read_u64::<LittleEndian>()?;
    dbg!(end_pos);

    let compression: Compression = reader.read_u32::<LittleEndian>()?.try_into()?;
    dbg!(compression);

    let meta_data = read_metadata(reader)?;
    dbg!(meta_data);

    // let library_version_major = reader.read_u32::<LittleEndian>()?;
    // let library_version_minor = reader.read_u32::<LittleEndian>()?;
    // dbg!(library_version_major);
    // dbg!(library_version_minor);

    let transform = read_transform(reader)?;
    dbg!(transform);

    let tree = read_tree(reader, compression)?;
    // dbg!(tree);
    // for v in tree.root_nodes[0].mask.iter() {

    // }

    // for idx in tree.root_nodes[0].mask.iter_ones() {
    //     let node_4 = read_node_header(reader, 4 /* 16 * 16 * 16 */, compression)?;

    //     let mut root_str = String::new();

    //     for (idx, v) in node_4.mask.iter().enumerate() {
    //         if idx % 16 == 0 {
    //             root_str += "\n";
    //         }

    //         if idx % (16 * 16) == 0 {
    //             root_str += "\n";
    //         }

    //         root_str += if *v {
    //             "-"
    //         } else {
    //             " "
    //         };
    //     }
    //     println!("{}", root_str);
    // }

    Ok(Grid {})
}

#[derive(Debug)]
struct Metadata {
    name: String,
    value: MetadataValue,
}

#[derive(Debug)]
enum MetadataValue {
    String(String),
    Vec3i(glam::IVec3),
    I64(i64),
    Bool(bool),
}

fn read_metadata<R: Read + Seek>(reader: &mut R) -> Result<Vec<Metadata>, ParseError> {
    let meta_data_count = reader.read_u32::<LittleEndian>()?;
    dbg!(meta_data_count);

    let mut meta_data = Vec::with_capacity(meta_data_count as usize);
    for _ in 0..meta_data_count {
        let name = read_name(reader)?;
        let data_type = read_name(reader)?;

        meta_data.push(Metadata {
            name: dbg!(name),
            value: match data_type.as_str() {
                "string" => MetadataValue::String(read_name(reader)?),
                "bool" => {
                    let len = reader.read_i32::<LittleEndian>()?;
                    let val = reader.read_u8()?;
                    dbg!(MetadataValue::Bool(val != 0))
                }
                "int64" => {
                    let len = reader.read_i32::<LittleEndian>()?;
                    let val = reader.read_i64::<LittleEndian>()?;
                    dbg!(MetadataValue::I64(val))
                }
                "vec3i" => {
                    let len = reader.read_i32::<LittleEndian>()?;
                    dbg!(MetadataValue::Vec3i(read_i_vec3(reader)?))
                }
                v => panic!("Invalid datatype {:?}", v),
            },
        })
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

#[derive(Debug)]
struct NodeHeader {
    child_mask: BitVec<u64, Lsb0>,
    value_mask: BitVec<u64, Lsb0>,
    data: Vec<f16>,
    log_2_dim: u32,
}

#[derive(Debug)]
struct Node3 {
    buffer: Vec<f16>,
    value_mask: BitVec<u64, Lsb0>,
    origin: glam::IVec3,
}

#[derive(Debug)]
struct Node4 {
    child_mask: BitVec<u64, Lsb0>,
    value_mask: BitVec<u64, Lsb0>,
    nodes: HashMap<u32, Node3>,
}

#[derive(Debug)]
struct Node5 {
    child_mask: BitVec<u64, Lsb0>,
    value_mask: BitVec<u64, Lsb0>,
    nodes: HashMap<u32, Node4>,
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

fn read_compressed<R: Read + Seek>(
    reader: &mut R,
    log_2_dim: u32,
    compression: Compression,
    value_mask: &BitSlice<u64, Lsb0>,
) -> Result<Vec<half::f16>, ParseError> {
    let linear_dim = (1 << (3 * log_2_dim)) as usize;
    let meta_data: NodeMetaData = reader.read_u8()?.try_into()?;

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

    let mut selection_mask = bitvec![u64, Lsb0; 0; linear_dim];

    if meta_data == NodeMetaData::MaskAndNoInactiveVals
        || meta_data == NodeMetaData::MaskAndOneInactiveVal
        || meta_data == NodeMetaData::MaskAndTwoInactiveVals
    {
        // let selection_mask = reader.read_u32::<LittleEndian>()?;
        reader.read_u64_into::<LittleEndian>(selection_mask.as_raw_mut_slice())?;
    }

    let data_size = match compression {
        Compression::ActiveMask if meta_data != NodeMetaData::NoMaskAndAllVals => {
            value_mask.count_ones()
        }
        _ => linear_dim,
    };

    let mut data = vec![0u16; data_size];
    reader.read_u16_into::<LittleEndian>(data.as_mut_slice())?;

    Ok(data.reinterpret_into())
}

fn read_node_header<R: Read + Seek>(
    reader: &mut R,
    log_2_dim: u32,
    compression: Compression,
) -> Result<NodeHeader, ParseError> {
    let linear_dim = (1 << (3 * log_2_dim)) as usize;

    let mut child_mask = bitvec![u64, Lsb0; 0; linear_dim];
    let mut value_mask = bitvec![u64, Lsb0; 0; linear_dim];
    reader.read_u64_into::<LittleEndian>(child_mask.as_raw_mut_slice())?;
    reader.read_u64_into::<LittleEndian>(value_mask.as_raw_mut_slice())?;

    let data = read_compressed(reader, log_2_dim, compression, value_mask.as_slice())?;

    Ok(NodeHeader {
        child_mask,
        value_mask,
        data,
        log_2_dim,
    })
}

#[derive(Debug)]
struct Tree {
    // origin: glam::IVec3
    root_nodes: Vec<Node5>,
}

fn read_tree<R: Read + Seek>(reader: &mut R, compression: Compression) -> Result<Tree, ParseError> {
    let expect_one_unk0 = reader.read_u32::<LittleEndian>()?;
    dbg!(expect_one_unk0);

    let root_node_background_value = reader.read_u32::<LittleEndian>()?;
    dbg!(root_node_background_value);

    let number_of_tiles = reader.read_u32::<LittleEndian>()?;
    dbg!(number_of_tiles);

    let number_of_root_nodes = reader.read_u32::<LittleEndian>()?;
    dbg!(number_of_root_nodes);

    let mut root_nodes = vec![];

    for n in 0..number_of_tiles {
        let vec = read_i_vec3(reader)?;
        let value = reader.read_u32::<LittleEndian>()?;
        let active = reader.read_u8()?;

        dbg!(vec);
    }

    for root_idx in 0..number_of_root_nodes {
        let origin = read_i_vec3(reader)?;
        dbg!(origin);

        let node_5 = read_node_header(reader, 5 /* 32 * 32 * 32 */, compression)?;
        let mut child_5 = HashMap::default();

        for idx in node_5.child_mask.iter_ones() {
            let node_4 = read_node_header(reader, 4 /* 16 * 16 * 16 */, compression)?;
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

    println!("Reasding data");

    for root_idx in 0..number_of_root_nodes {
        let node_5 = &mut root_nodes[root_idx as usize];
        for idx in node_5.child_mask.iter_ones() {
            let mut node_4 = node_5.nodes.get_mut(&(idx as u32)).unwrap();

            for idx in node_4.child_mask.iter_ones() {
                let mut node_3 = node_4.nodes.get_mut(&(idx as u32)).unwrap();

                let linear_dim = (1 << (3 * 3)) as usize;
                let mut value_mask = bitvec![u64, Lsb0; 0; linear_dim];
                reader.read_u64_into::<LittleEndian>(value_mask.as_raw_mut_slice())?;

                // println!("{:?}", reader.stream_position());
                let data = read_compressed(reader, 3, compression, value_mask.as_slice())?;

                node_3.buffer = data;
            }
        }
    }

    Ok(Tree {
        root_nodes,
        // origin
    })
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
            _ => return Err(ParseError::InvalidCast),
        })
    }
}

#[derive(Debug, Copy, Clone)]
enum Compression {
    None,
    Zip,
    ActiveMask,
    Blosc,
}

impl TryFrom<u32> for Compression {
    type Error = ParseError;

    fn try_from(v: u32) -> Result<Compression, ParseError> {
        Ok(match v {
            0 => Compression::None,
            1 => Compression::Zip,
            2 => Compression::ActiveMask,
            3 => Compression::Blosc,
            _ => return Err(ParseError::InvalidCast),
        })
    }
}

fn read_vdb<R: Read + Seek>(reader: &mut R) -> Result<(), ParseError> {
    let magic = reader.read_u64::<LittleEndian>()?;
    if magic == 0x2042445600000000 {
        return Err(ParseError::MagicMismatch);
    }

    let file_version = reader.read_u32::<LittleEndian>()?;
    dbg!(file_version);

    let library_version_major = reader.read_u32::<LittleEndian>()?;
    let library_version_minor = reader.read_u32::<LittleEndian>()?;
    dbg!(library_version_major);
    dbg!(library_version_minor);

    let grid_offsets = reader.read_u8()?;
    dbg!(grid_offsets);

    let guid = read_string(reader, 36)?;
    dbg!(guid);

    // jb-todo: proper meta-data parsing
    let meta_data = read_metadata(reader)?;
    dbg!(meta_data);

    let grid_count = reader.read_u32::<LittleEndian>()?;
    dbg!(grid_count);

    let grid = read_grid(reader)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let f = File::open("C:/Users/Jasper/Downloads/armadillo.vdb-1.0.0/armadillo.vdb")?;
    // let f = File::open("C:/Users/Jasper/Downloads/cube.vdb-1.0.0/cube.vdb")?;
    let mut reader = BufReader::new(f);

    dbg!(read_vdb(&mut reader))?;
    // let mut child_mask = bitvec![u64, Lsb0; 0; 2];
    // child_mask.set(0, true);
    // child_mask.set(1, true);
    // child_mask.set(3, true); // oob

    Ok(())
}
