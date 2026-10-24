# End-to-end encrypted fine-tuning experiment

**Note: The optimal runtime can be achieved by stream-level parallelism but performance varies from type of GPU and computation. Here we set OpenMP thread to one which is bundled with the single stream for a stable performance.**

## Requirements and installation

Use the CUDA Toolkit, compiler, CMake, NTL and GMP dependencies described in
[GPU_core](../GPU_core/README.md#requirements-and-installation). This directory
builds its own bundled GPU implementation and RBOOT GPU sources.

## Run

From the repository root, for an H200 or H100 GPU (`sm_90`):

```bash
cd e2e_experiment_GPU
CUDA_VISIBLE_DEVICES=0 bash run.sh \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  > results.csv 2> run.log
```

Set `CMAKE_CUDA_ARCHITECTURES` to the architecture of the target GPU. If needed,
pass `-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc` to `run.sh`.