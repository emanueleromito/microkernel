# microkernel

Quantization-aware training library in C++ and CUDA.

## Core Idea

Custom CUDA kernels for low-precision neural network training.

## Modules

```
Tensor          — N-dim array, device-aware, carries dtype
DeviceBuffer    — RAII wrapper around cudaMalloc/cudaFree
Kernels         — CUDA kernels per op, per precision
Quantize        — Fake quant nodes, scale/zero-point calibration, STE backward
Autograd        — Dynamic computation graph, reverse-mode AD
Layers          — Linear, Conv2d, ReLU, BatchNorm (each with quantized variants)
Optim           — SGD, Adam
Data            — MNIST/CIFAR-10 loader, batching
```


## Architecture

```
User code
  │
  ▼
Layers (forward / backward)
  │
  ▼
Autograd (graph, gradient flow, STE)
  │
  ▼
Tensor (shape, strides, dtype, device)
  │
  ├──► Quantize (fake_quantize, calibrate, pack/unpack)
  │
  ├──► CUDA Kernels          ├──► CPU Backend
  │    fp32 matmul (naive)    │    naive loops
  │    fp32 matmul (tiled)    │    reference for testing
  │    fp16 matmul            │
  │    int8 matmul            │
  │    int4 packed matmul     │
  │    elementwise, reduce    │
  │    conv2d                 │
  │
  ▼
DeviceBuffer (cudaMalloc, cudaFree, host↔device)
```


## File Layout

```
microkernel/
├── CMakeLists.txt
├── include/microkernel/
│   ├── tensor.h
│   ├── device_buffer.h
│   ├── autograd.h
│   ├── quantize.h
│   ├── layers/
│   └── optim/
├── src/
│   ├── tensor.cpp
│   ├── autograd.cpp
│   ├── quantize.cpp
│   └── data/
├── kernels/
│   ├── matmul_fp32.cu
│   ├── matmul_fp16.cu
│   ├── matmul_int8.cu
│   ├── matmul_int4.cu
│   ├── elementwise.cu
│   ├── reduce.cu
│   ├── conv2d.cu
│   └── fake_quantize.cu
├── tests/
└── examples/
    ├── mnist_fp32.cpp
    ├── mnist_qat_int8.cpp
    └── cifar10_qat_int4.cpp
```



## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest
```

Requires: CMake 3.18+, CUDA 11+, C++17, Google Test.