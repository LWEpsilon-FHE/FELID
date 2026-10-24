# CPU implementation of the core algorithms and tests

## Requirements and installation

A C++17 compiler, CMake 3.20 or newer, OpenMP, NTL and GMP are required.
On Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential cmake git libntl-dev libgmp-dev
```

The modified SEAL-4.1-bs library in `thirdparty/SEAL-4.1-bs` is built automatically with this
project.

## Build and run

From the repository root:

```bash
cd CPU_core
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target core_test -j 4

./build/test --list
./build/test all
```
