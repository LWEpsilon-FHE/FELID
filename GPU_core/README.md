# GPU implementation of the core algorithms and tests

**Note: The optimal runtime can be achieved by stream-level parallelism but performance varies from type of GPU and computation. Here we set OpenMP thread to one which is bundled with the single stream for a stable performance.**
## Requirements and installation

The tested toolchain uses CUDA Toolkit 11.8, GCC/G++ 11.4, OpenMP and CMake 3.24
or newer. An NVIDIA GPU and a compatible NVIDIA driver are required.
On Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential gcc-11 g++-11 git libntl-dev libgmp-dev python3-venv
python3 -m venv ~/.venvs/felid-build
source ~/.venvs/felid-build/bin/activate
python -m pip install "cmake>=3.24,<4"
cmake --version
nvcc --version
nvidia-smi
```

Keep the build environment activated. Make the CUDA Toolkit's `bin` directory
available on `PATH`, or pass `-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc` to CMake.
The bundled PhantomFHE library is built automatically.

## Build and run

From the repository root, for an H200 or H100 GPU (`sm_90`):

```bash
cd GPU_core
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build --target core_test -j 4
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
CUDA_VISIBLE_DEVICES=0 ./build/test all
```

