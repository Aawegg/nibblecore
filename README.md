# NibbleCore

**4-bit quantization kernels for Apple Silicon, benchmarked against llama.cpp.**

From-scratch Q4_0 quantization/dequantization with NEON SIMD optimization, a GGUF model file parser, and a head-to-head comparison against llama.cpp's reference implementation. Built for and tested on Apple M4.

## Results (M4 MacBook Air)

### Dequantization: 3.4x faster than llama.cpp

Tested on real weights from SmolLM2-135M (884,736 weights from `blk.0.ffn_down.weight`):

| Implementation | Time (1K iters) | Throughput | vs llama.cpp |
|---------------|----------------|-----------|-------------|
| llama.cpp scalar (reference) | 194.96 ms | 16.91 GB/s | baseline |
| NibbleCore scalar | 176.87 ms | 18.63 GB/s | 1.10x |
| NibbleCore NEON v1 | 63.92 ms | 51.56 GB/s | **3.05x** |
| NibbleCore NEON v2 | 57.02 ms | 57.80 GB/s | **3.42x** |

All implementations produce bit-identical output (max diff = 0.00).

### Quantized Dot Product: The Real Trick

In production, llama.cpp never dequantizes. It computes dot products directly on packed 4-bit data:

| Method | Time (1K iters) | Speedup |
|--------|----------------|---------|
| Dequantize + float dot | 116.76 ms | baseline |
| Q4_0 * Q8_0 scalar | 52.98 ms | 2.2x |
| Q4_0 * Q8_0 NEON (SDOT) | 24.17 ms | **4.8x** |

### Compression

| Metric | Value |
|--------|-------|
| Original (float32) | 128 bytes / 32 weights |
| Quantized (Q4_0) | 18 bytes / 32 weights |
| Compression ratio | 7.1x |
| Space saved | 85.9% |
| Reconstruction MSE | 0.0016 |

A 7B parameter model goes from 26 GB to 3.7 GB — small enough for an M4 MacBook Air.

## What's Inside

### `main.cpp` — Quantization Kernel + Benchmark
The core Q4_0 quantization and dequantization implementation with both scalar and NEON SIMD paths. Benchmarks throughput on 1M floats.

### `demo.cpp` — Visual Step-by-Step
Color-coded terminal visualization that shows quantization happening in real time: the scale factor calculation, bit packing, and reconstruction with error analysis.

### `gguf.hpp` — GGUF Format Parser
Parses the GGUF file format (used by llama.cpp for model storage). Reads headers, metadata, and tensor descriptors from real model files.

### `loader.cpp` — Real Model Loader
Opens a GGUF model file, displays architecture details (layer count, embedding size, vocab), lists all tensors, and dequantizes real weights with statistics and histogram.

### `compare.cpp` — Head-to-Head vs llama.cpp
Benchmarks four dequantization implementations and three dot product approaches on real model data. Includes llama.cpp's exact scalar implementation from `ggml-quants.c` for fair comparison.

## Build

Requires: macOS with Apple Silicon (M1/M2/M3/M4), Xcode Command Line Tools.

```bash
make          # build everything
make run      # Sprint 1: quantization benchmark
make watch    # Sprint 1: visual demo
make load     # Sprint 3: load a real GGUF model
make versus   # Sprint 4: head-to-head vs llama.cpp
```

## Quick Start

```bash
git clone https://github.com/Aawegg/nibblecore.git
cd nibblecore
make

# Download a small model to test with (~90MB)
mkdir -p models
curl -L -o models/smollm2-135m-q4_0.gguf \
  'https://huggingface.co/QuantFactory/SmolLM2-135M-GGUF/resolve/main/SmolLM2-135M.Q4_0.gguf'

# Run the comparison against llama.cpp
./compare models/smollm2-135m-q4_0.gguf
```

## The Q4_0 Format

Each block stores 32 weights in 18 bytes:

```
┌────────────┬──────────────────────────────────┐
│ scale (f16)│ 16 bytes: 32 weights @ 4 bits    │
│  2 bytes   │ (two weights packed per byte)     │
└────────────┴──────────────────────────────────┘
```

Packing: `byte = low_nibble | (high_nibble << 4)`
Unpacking: `low = byte & 0x0F`, `high = byte >> 4`
Dequantize: `float = (quantized - 8) * scale`

This is the same format used by llama.cpp (`block_q4_0` in `ggml-quants.c`).

## Key Technical Details

- **NEON SIMD**: ARM's 128-bit vector instructions process 4 floats per cycle. The advanced dequantization path uses `vmovl` (widen) chains to go from packed uint8 → int16 → int32 → float32 entirely in registers.

- **SDOT instruction**: The quantized dot product uses `vdotq_s32`, which performs 4 int8 multiply-accumulates per lane (16 total per instruction). This is why quantized inference is fast — the math stays in integer domain until the final scale multiplication.

- **BlockQ4_0 layout**: The interleaved packing (values at positions `i` and `i+16` share a byte) matches llama.cpp's format and improves cache access patterns during dequantization.

## License

MIT
