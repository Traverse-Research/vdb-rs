☁️ `vdb-rs`
========
[![Actions Status](https://github.com/Traverse-Research/vdb-rs/workflows/Continuous%20integration/badge.svg)](https://github.com/Traverse-Research/vdb-rs/actions)
[![Latest version](https://img.shields.io/crates/v/vdb-rs.svg)](https://crates.io/crates/vdb-rs)
[![Documentation](https://docs.rs/vdb-rs/badge.svg)](https://docs.rs/vdb-rs)
[![Lines of code](https://tokei.rs/b1/github/Traverse-Research/vdb-rs)](https://github.com/Traverse-Research/vdb-rs)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
[![Contributor Covenant](https://img.shields.io/badge/contributor%20covenant-v1.4%20adopted-ff69b4.svg)](../master/CODE_OF_CONDUCT.md)

[![Banner](banner.png)](https://traverseresearch.nl)

This crate provides a rust native implementation of the VDB file format, following the original [OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb) implementation.

- [Documentation](https://docs.rs/vdb-rs)

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
vdb-rs = "0.2.1"
```

This crate currently only supports VDB reading and parsing of a relatively large section of the VDB test assets, while it currently
only supports reading the data an nothing more, the longer term goal for this is to reach feature parity with the C++ OpenVDB crate.
Implementation of features however is use-case limited, so contributions in areas that are missing are welcome.

# Known missing features

1. Multi-pass I/O (`PointDataGrid`)
1. VDB Writing
1. Older OpenVDB versions
1. DDA tracing (with example)
1. Delay loading

# Broken files

These are test files from the OpenVDB website; https://www.openvdb.org/download/. Most file seem to be loading correctly
and displaying correctly in the `bevy` example that's provided with this library.

Most of these errors seem to be related to the lack of Multi-Pass I/O, though most need to be investigated.

1. no visuals: "smoke2.vdb-1.0.0/smoke2.vdb"
1. parse error: "torus.vdb-1.0.0/torus.vdb" InvalidNodeMetadata
1. parse erorr: "venusstatue.vdb-1.0.0/venusstatue.vdb" InvalidNodeMetadata
1. parse error: "boat_points.vdb-1.0.0/boat_points.vdb" InvalidCompression
1. parse error: "bunny_points.vdb-1.0.0/bunny_points.vdb" InvalidCompression
1. parse error: "sphere_points.vdb-1.0.0/sphere_points.vdb" InvalidCompression
1. parse error: "waterfall_points.vdb-1.0.0/waterfall_points.vdb" InvalidCompression
