# microkernel — Implementation Plan

## Phase 1: Foundation (CPU only)

- [X] Tensor class (fp32, shape, strides, fill, at)
- [X] CMakeLists.txt + Google Test setup
- [X] Tensor tests
- [ ] Tensor operators: +, -, *, element-wise
- [ ] Tensor reshape, transpose (views, no copy)

## Phase 2: CUDA Memory

- [ ] DeviceBuffer (RAII wrapper, cudaMalloc/cudaFree)
- [ ] Host ↔ Device transfers
- [ ] CUDA_CHECK error macro
- [ ] Tensor device awareness (CPU/CUDA), `.to(device)`

## Phase 3: CUDA Kernels (fp32)

- [ ] Elementwise kernels (add, mul, relu)
- [ ] Matmul naive (one thread per output element)
- [ ] Matmul tiled
- [ ] Matmul shared memory
- [ ] Reduction kernels (sum, max, mean)

## Phase 4: Autograd

- [ ] GradFn node, computation graph
- [ ] Reverse-mode AD (topological sort + backward walk)
- [ ] Gradient checking (numerical vs autograd)

## Phase 5: Training (fp32)

- [ ] Linear, ReLU, Softmax layers
- [ ] Cross-entropy loss
- [ ] SGD optimizer
- [ ] MNIST data loader
- [ ] Train a 2-layer MLP on MNIST

## Phase 6: Mixed Precision

- [ ] fp16 storage in Tensor
- [ ] fp16 CUDA kernels
- [ ] Mixed-precision training (fp16 forward, fp32 gradients)

## Phase 7: Quantization-Aware Training

- [ ] Fake quantization nodes (quantize → clamp → dequantize)
- [ ] Straight-through estimator backward
- [ ] Scale/zero-point calibration
- [ ] QAT training loop on MNIST

## Phase 8: Low-Precision Kernels

- [ ] int8 matmul kernel
- [ ] int4 packed matmul kernel (2 values per byte)
- [ ] Grouped quantization
- [ ] Benchmark against cuBLAS

## Phase 9: Convolution

- [ ] Conv2d kernel (fp32)
- [ ] Conv2d quantized variants
- [ ] MaxPool2d
- [ ] CIFAR-10 data loader
- [ ] Train a small CNN on CIFAR-10 with QAT

## Phase 10: Profiling & Optimization

- [ ] nsight-compute profiling
- [ ] Occupancy analysis
- [ ] Fused kernels (linear + relu + bias)
- [ ] Memory pool allocator
- [ ] CUDA streams for overlap