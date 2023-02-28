# `vdb-rs`

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
