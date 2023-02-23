use std::fs::File;
use std::io::BufReader;
use std::io::{Seek, Read, SeekFrom};
use std::error::Error;
use byteorder::{ReadBytesExt, WriteBytesExt, BigEndian, LittleEndian};
use std::ffi::{OsStr, OsString};

#[derive(thiserror::Error, Debug)]
enum ParseError {
    #[error("Magic bytes mismatched")]
    MagicMismatch,
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

    let compression = reader.read_u32::<LittleEndian>()?;
    dbg!(compression);

    let meta_data = read_metadata(reader)?;
    dbg!(meta_data);

    // let library_version_major = reader.read_u32::<LittleEndian>()?;
    // let library_version_minor = reader.read_u32::<LittleEndian>()?;
    // dbg!(library_version_major);
    // dbg!(library_version_minor);

    let transform = read_transform(reader)?;
    dbg!(transform);

    let tree = read_tree(reader)?;
    dbg!(tree);

    Ok(Grid {} )
}

#[derive(Debug)]
struct Metadata {
    name: String,
    value: MetadataValue
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
                },
                "int64" => {
                    let len = reader.read_i32::<LittleEndian>()?;
                    let val = reader.read_i64::<LittleEndian>()?;
                    dbg!(MetadataValue::I64(val))
                },
                "vec3i" => {
                    let len = reader.read_i32::<LittleEndian>()?;
                    dbg!(MetadataValue::Vec3i(read_i_vec3(reader)?))
                },
                v => panic!("Invalid datatype {:?}", v),
            }
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
    }
}

fn read_transform<R: Read + Seek>(reader: &mut R) -> Result<Map, ParseError> {
    let name = read_name(reader)?;

    Ok(match name.as_str() {
        "UniformScaleMap" => {
            Map::UniformScaleMap {
                scale_values: read_d_vec3(reader)?,
                voxel_size: read_d_vec3(reader)?,
                scale_values_inverse: read_d_vec3(reader)?,
                inv_scale_sqr: read_d_vec3(reader)?,
                inv_twice_scale: read_d_vec3(reader)?,
            }
        },
        v => panic!("Not supported {}", v)
    })
}

struct NodeHeader {
    mask: Vec<u64>,
    value_mask: Vec<u64>,
    data: Vec<u16>,
    compression_mode: u8
}

fn read_node_header<R: Read + Seek>(reader: &mut R, log_2_dim: u32) -> Result<NodeHeader, ParseError> {
    dbg!(log_2_dim);
    let dim = (1 << log_2_dim) as usize;
    let linear_dim = dim * dim * dim;
    let linear_bit_count = linear_dim / 64; // bits stored as u64

    let mut mask = vec![0u64; linear_bit_count];
    let mut value_mask = vec![0u64; linear_bit_count];
    reader.read_u64_into::<LittleEndian>(mask.as_mut_slice())?;
    reader.read_u64_into::<LittleEndian>(value_mask.as_mut_slice())?;

    let data_size = if true { 
        value_mask.iter().map(|v| v.count_ones() as usize).sum()
    } else {
        linear_dim
    };

    let compression_mode = reader.read_u8()?;
    dbg!(compression_mode);

    let mut data = vec![0u16; data_size]; // half float
    reader.read_u16_into::<LittleEndian>(data.as_mut_slice())?;

    Ok(NodeHeader {
        mask,
        value_mask,
        data,
        compression_mode
    })
}

fn read_tree<R: Read + Seek>(reader: &mut R) -> Result<(), ParseError> {
    let expect_one_unk0 = reader.read_u32::<LittleEndian>()?;
    dbg!(expect_one_unk0);

    let root_node_background_value = reader.read_u32::<LittleEndian>()?;
    dbg!(root_node_background_value);

    let number_of_tiles = reader.read_u32::<LittleEndian>()?;
    dbg!(number_of_tiles);

    let number_of_5_nodes = reader.read_u32::<LittleEndian>()?;
    dbg!(number_of_5_nodes);

    let origin = read_i_vec3(reader)?;
    dbg!(origin);

    let node_5_header = read_node_header(reader, 5 /* 32 * 32 * 32 */)?;
    // dbg!(node_5_header);

    for (idx, word) in node_5_header.mask.iter().enumerate() {
        let mut word = *word;
        while word != 0 {
            // let bit_index = idx as u32 * 64 + word.trailing_zeros();

            let node_4_header = read_node_header(reader, 4 /* 16 * 16 * 16 */)?;
            for (idx, word) in node_4_header.mask.iter().enumerate() {
                let mut word = *word;
                while word != 0 {
                    // let bit_index = idx as u32 * 64 + word.trailing_zeros();

                    
                    let mut data = vec![0u64; 8]; // mask
                    reader.read_u64_into::<LittleEndian>(data.as_mut_slice())?;

                    word &= word - 1;
                }
            }

            word &= word - 1;
        }
    }

    Ok(())
}

fn read_vdb<R: Read + Seek>(reader: &mut R) -> Result<(), ParseError> {
    let magic = reader.read_u64::<LittleEndian>()?;
    if  magic == 0x2042445600000000 {
        return Err(ParseError::MagicMismatch)
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

    let grid = read_grid(reader);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let f = File::open("C:/Users/Jasper/Downloads/armadillo.vdb-1.0.0/armadillo.vdb")?;
    let mut reader = BufReader::new(f);

    read_vdb(&mut reader)?;

    Ok(())
}
