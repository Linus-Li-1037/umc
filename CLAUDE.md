# Project: Unstructured data compression

Develop compression routines for unstructured scientific data.

## Build

- `mkdir build && cd build && cmake .. && make -j$(nproc)` — full build

## test
compression: ./build/test/test_compress $data_dir $variable $coordinate $connectivity $from_edge_or_not $dimension $compression_option $reorder_option $num_timesteps $abs_eb_or_not $eb + additional parameters based on compression options
decompression: ./build/test/test_decompress $data_dir $variable $coordinate $connectivity $from_edge_or_not $dimension $compression_option $reorder_option $num_timesteps + additional parameters based on compression options

- ./build/test/test_compress data/Tsunami/ height coordinates.dat connectivity.dat 1 2 2 1 1 0 1e-2 4
- ./build/test/test_decompress data/Tsunami/ height coordinates.dat connectivity.dat 1 2 2 1 1
- python python_hierarchy/generate_centroid_triangulation.py <mesh_nc> data/Tsunami/   # run once
- ./build/test/test_parameterization_tsunami data/Tsunami/
Compressed file is data/Tsunami/height.dat.0.umc. 
Decompressed file is data/Tsunami/height.dat.0.umc.out

- ./build/test/test_compress data/Katrina/ attr0 coordinates.dat connectivity.dat 0 3 4 1 3 0 1e-2 4 10
- ./build/test/test_decompress data/Katrina/ attr0 coordinates.dat connectivity.dat 0 3 4 1 3 10
Compressed files are data/Katrina/attr0.dat.0.umc, data/Katrina/attr0.dat.1.umc, data/Katrina/attr0.dat.2.umc. 
Decompressed files are data/Katrina/attr0.dat.0.umc.out, data/Katrina/attr0.dat.1.umc.out, data/Katrina/attr0.dat.2.umc.out.


### Dependencies

- CMake
- C++
- GSL
- ZFP
- METIS
- FTK
- SZ3
- ZSTD

## Structure

- `build`       		— build folder
- `data`				— data folder for Katrina and Trunami
- `env`					— running experiments on a cluster. Please ingore for now.
- `external`			— code and installation of dependencies
- `include` 			— headers files and implementations
- `parallel_src`		- source code with parallelization. Please ignore for now.
- `python_hierarchy`	- python implementation of building hierarchical meshes. Will work on it later.
- `test`				- example source code for basic tests

## Code Conventions

### Memory and data layout
- All distributed arrays use row-major layout. 

### General C++ style
- Use `const` and pass by reference wherever possible.

## Important notes
- Please do not touch `env`, `external`, and `parallel_src` at this stage.

