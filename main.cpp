// ============================================================================
// NibbleCore: 4-bit Quantization Kernel for Apple M4
// ============================================================================
//
// THE BIG IDEA
// ------------
// Neural network weights are stored as 32-bit floats (4 bytes each).
// A 7-billion parameter model = 7B x 4 bytes = 28 GB.
// Your M4 Air has 16-24 GB of unified memory. That model won't fit.
//
// 4-bit quantization stores each weight as a 4-bit integer (0.5 bytes).
// 7B x 0.5 bytes = 3.5 GB. Now it fits, with room to spare.
//
// The trade-off: you lose some precision. But research shows that 4-bit
// quantized models retain 95-99% of their original quality for most tasks.
//
// HOW IT WORKS
// ------------
// We process weights in blocks of 32. For each block:
//   1. Find the largest absolute value (the "scale")
//   2. Divide all 32 values by the scale -> normalized to [-1, 1]
//   3. Map from [-1, 1] to [0, 15] (4-bit range)
//   4. Pack two 4-bit values into each byte (50% space savings on top)
//
// This is the exact same format used by llama.cpp (Q4_0).
// ============================================================================

#include <cstdint>     // uint8_t, uint16_t -- exact-width integer types
#include <cstdlib>     // rand(), RAND_MAX
#include <cstring>     // memset
#include <cmath>       // fabsf, roundf
#include <cstdio>      // printf
#include <vector>
#include "benchmark.hpp"

// On macOS/ARM, this gives us the float16 type (half precision).
// float16 is 2 bytes instead of float's 4 bytes -- another space saving
// just for storing the scale factor.
#include <arm_neon.h>

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// This struct stores 32 quantized weights.
//
// Memory layout:
//   - scale: 2 bytes (float16 -- half precision float)
//   - qs:   16 bytes (32 weights packed into 4 bits each, so 32/2 = 16 bytes)
//   Total: 18 bytes for 32 weights
//
// Compare to original: 32 weights x 4 bytes = 128 bytes
// Compression ratio: 128 / 18 = 7.1x
//
// This is identical to llama.cpp's block_q4_0 format.
struct BlockQ4_0 {
    float16_t scale;    // The "ruler" -- multiply quantized values by this to
                        // get back approximate original values
    uint8_t qs[16];     // 32 weights, packed 2 per byte (4 bits each)
};

// Verify our struct is exactly 18 bytes with no padding.
// If the compiler adds hidden padding bytes, our memory math breaks.
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be 18 bytes");

// How many floats go into each block
constexpr int BLOCK_SIZE = 32;

// ============================================================================
// SCALAR QUANTIZATION (the simple, understandable version)
// ============================================================================
// This does the quantization using plain C++ -- no SIMD tricks.
// It's slower but easy to follow. We'll compare it to the NEON version.

void quantize_row_scalar(const float* input, BlockQ4_0* output, int num_floats) {
    // Process 32 floats at a time
    int num_blocks = num_floats / BLOCK_SIZE;

    for (int block = 0; block < num_blocks; block++) {
        const float* src = input + block * BLOCK_SIZE;
        BlockQ4_0& dst = output[block];

        // STEP 1: Find the absolute max value in this block of 32.
        //
        // Why? We need to know the "range" of values so we can scale them
        // into 4 bits (0-15). If the max is 3.7, we know all values are
        // between -3.7 and +3.7.
        float amax = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float a = fabsf(src[i]);
            if (a > amax) amax = a;
        }

        // STEP 2: Calculate the scale factor.
        //
        // We want to map the range [-amax, +amax] to [0, 15].
        // The center point is 8 (since 0-15 has 16 values, 8 is the middle).
        //
        //   quantized = (original / scale) + 8
        //
        // So: scale = amax / 8
        //
        // If amax is 0 (all weights are zero), scale is 0 and everything
        // stays zero. No division by zero.
        float scale = amax / 8.0f;
        float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

        // Store scale as float16 (half precision) to save space.
        // On ARM, float16_t (__fp16) supports implicit conversion from float.
        dst.scale = static_cast<float16_t>(scale);

        // STEP 3: Quantize each value and pack pairs into bytes.
        //
        // For each pair of values (i and i+16):
        //   - Normalize: value / scale -> range roughly [-8, 8]
        //   - Shift to unsigned: add 8 -> range roughly [0, 16]
        //   - Clamp to [0, 15] (4-bit range)
        //   - Pack: first value goes in the LOW 4 bits, second in the HIGH 4 bits
        //
        // Why i and i+16? This interleaving pattern is what llama.cpp uses.
        // It improves cache access patterns during dequantization.
        for (int i = 0; i < 16; i++) {
            // Quantize value at position i (goes into low nibble)
            float v0 = src[i] * inv_scale + 8.0f;
            uint8_t q0 = static_cast<uint8_t>(roundf(v0));
            if (q0 > 15) q0 = 15;

            // Quantize value at position i+16 (goes into high nibble)
            float v1 = src[i + 16] * inv_scale + 8.0f;
            uint8_t q1 = static_cast<uint8_t>(roundf(v1));
            if (q1 > 15) q1 = 15;

            // THE PACKING:
            // q0 occupies bits 0-3 (the low nibble)
            // q1 is shifted left 4 bits to occupy bits 4-7 (the high nibble)
            // OR them together -> one byte holds two weights
            dst.qs[i] = q0 | (q1 << 4);
        }
    }
}

// ============================================================================
// SCALAR DEQUANTIZATION (reverse the process)
// ============================================================================
// Takes packed 4-bit values and reconstructs approximate floats.

void dequantize_row_scalar(const BlockQ4_0* input, float* output, int num_floats) {
    int num_blocks = num_floats / BLOCK_SIZE;

    for (int block = 0; block < num_blocks; block++) {
        const BlockQ4_0& src = input[block];
        float* dst = output + block * BLOCK_SIZE;

        // Convert float16 scale back to float32
        float scale = static_cast<float>(src.scale);

        for (int i = 0; i < 16; i++) {
            // UNPACKING:
            // Low nibble:  mask with 0x0F (binary 00001111) to get bits 0-3
            // High nibble: shift right 4 to get bits 4-7 into bits 0-3
            uint8_t packed = src.qs[i];
            uint8_t q0 = packed & 0x0F;        // low nibble -> value at position i
            uint8_t q1 = (packed >> 4) & 0x0F;  // high nibble -> value at position i+16

            // Reverse the quantization:
            //   original = (quantized - 8) * scale
            dst[i]      = (static_cast<float>(q0) - 8.0f) * scale;
            dst[i + 16] = (static_cast<float>(q1) - 8.0f) * scale;
        }
    }
}

// ============================================================================
// NEON SIMD QUANTIZATION (the fast version)
// ============================================================================
//
// NEON is ARM's SIMD (Single Instruction, Multiple Data) extension.
// Instead of processing one float at a time, NEON processes 4 floats
// at once using 128-bit registers.
//
// Your M4 chip has dedicated NEON hardware -- these aren't emulated,
// they're real silicon paths that do 4 operations in the same time as 1.

void quantize_row_neon(const float* input, BlockQ4_0* output, int num_floats) {
    int num_blocks = num_floats / BLOCK_SIZE;

    for (int block = 0; block < num_blocks; block++) {
        const float* src = input + block * BLOCK_SIZE;
        BlockQ4_0& dst = output[block];

        // STEP 1: Find absolute max using NEON.
        //
        // vld1q_f32:  loads 4 floats into a 128-bit NEON register
        // vabsq_f32:  takes absolute value of all 4 floats at once
        // vmaxq_f32:  takes element-wise max of two registers
        //
        // We process all 32 floats in groups of 4 (8 iterations),
        // keeping a running max across all groups.
        float32x4_t vmax = vdupq_n_f32(0.0f);  // initialize max = [0,0,0,0]
        for (int i = 0; i < BLOCK_SIZE; i += 4) {
            float32x4_t v = vld1q_f32(src + i);        // load 4 floats
            float32x4_t va = vabsq_f32(v);             // abs of all 4
            vmax = vmaxq_f32(vmax, va);                 // running max
        }

        // vmaxvq_f32: reduces the 4-element vector to a single max value
        // (horizontal max -- compares all 4 lanes and returns the biggest)
        float amax = vmaxvq_f32(vmax);

        float scale = amax / 8.0f;
        float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;
        dst.scale = static_cast<float16_t>(scale);

        // STEP 2: Quantize using NEON.
        //
        // vdupq_n_f32:    broadcasts a single value to all 4 lanes [8,8,8,8]
        // vmulq_f32:      multiplies 4 floats at once
        // vaddq_f32:      adds 4 floats at once
        // vcvtnq_u32_f32: converts 4 floats to 4 unsigned ints (round-to-nearest)
        // vminq_u32:      clamps to max value (element-wise min)
        //
        // We quantize 4 values per iteration instead of 1.
        float32x4_t vscale = vdupq_n_f32(inv_scale);
        float32x4_t voffset = vdupq_n_f32(8.0f);
        uint32x4_t vclamp = vdupq_n_u32(15);

        for (int i = 0; i < 16; i += 4) {
            // Quantize 4 values from the first half (positions i..i+3)
            float32x4_t v0 = vld1q_f32(src + i);
            v0 = vaddq_f32(vmulq_f32(v0, vscale), voffset);
            uint32x4_t q0 = vcvtnq_u32_f32(v0);
            q0 = vminq_u32(q0, vclamp);

            // Quantize 4 values from the second half (positions i+16..i+19)
            float32x4_t v1 = vld1q_f32(src + i + 16);
            v1 = vaddq_f32(vmulq_f32(v1, vscale), voffset);
            uint32x4_t q1 = vcvtnq_u32_f32(v1);
            q1 = vminq_u32(q1, vclamp);

            // Pack: q0 in low nibble, q1 shifted into high nibble, OR together.
            // NEON lane indices must be compile-time constants, so we store
            // to a temporary array instead of extracting lane-by-lane.
            uint32_t tmp0[4], tmp1[4];
            vst1q_u32(tmp0, q0);
            vst1q_u32(tmp1, q1);
            for (int j = 0; j < 4; j++) {
                dst.qs[i + j] = static_cast<uint8_t>(tmp0[j])
                              | static_cast<uint8_t>(tmp1[j] << 4);
            }
        }
    }
}

// NEON dequantization
void dequantize_row_neon(const BlockQ4_0* input, float* output, int num_floats) {
    int num_blocks = num_floats / BLOCK_SIZE;

    for (int block = 0; block < num_blocks; block++) {
        const BlockQ4_0& src = input[block];
        float* dst = output + block * BLOCK_SIZE;

        float scale = static_cast<float>(src.scale);
        float32x4_t vscale = vdupq_n_f32(scale);
        float32x4_t voffset = vdupq_n_f32(8.0f);

        for (int i = 0; i < 16; i += 4) {
            // Unpack 4 bytes -> 4 low nibbles and 4 high nibbles.
            // Load into a plain array, then move to NEON register.
            uint32_t tmp0[4], tmp1[4];
            for (int j = 0; j < 4; j++) {
                tmp0[j] = src.qs[i + j] & 0x0F;
                tmp1[j] = (src.qs[i + j] >> 4) & 0x0F;
            }
            uint32x4_t q0 = vld1q_u32(tmp0);
            uint32x4_t q1 = vld1q_u32(tmp1);

            // Convert to float and reverse quantization
            float32x4_t f0 = vcvtq_f32_u32(q0);
            f0 = vmulq_f32(vsubq_f32(f0, voffset), vscale);
            vst1q_f32(dst + i, f0);

            float32x4_t f1 = vcvtq_f32_u32(q1);
            f1 = vmulq_f32(vsubq_f32(f1, voffset), vscale);
            vst1q_f32(dst + i + 16, f1);
        }
    }
}

// ============================================================================
// QUALITY MEASUREMENT
// ============================================================================
// Quantization loses precision. How much? We measure with two metrics:
//
// MSE (Mean Squared Error): average of (original - reconstructed)^2
//   Lower = better. 0 = perfect reconstruction.
//
// Max Error: the single worst-case deviation.
//   Tells you how bad the worst approximation is.

struct QualityMetrics {
    double mse;
    float max_error;
};

QualityMetrics measure_quality(const float* original, const float* reconstructed, int n) {
    double sum_sq = 0.0;
    float max_err = 0.0f;

    for (int i = 0; i < n; i++) {
        float err = fabsf(original[i] - reconstructed[i]);
        sum_sq += static_cast<double>(err) * err;
        if (err > max_err) max_err = err;
    }

    return { sum_sq / n, max_err };
}

// ============================================================================
// MAIN: Run everything and show results
// ============================================================================

int main() {
    // Number of floats to process.
    // 1M floats = 4 MB of data (at 4 bytes each).
    // Real models have billions, but 1M is enough to measure speed reliably.
    constexpr int NUM_FLOATS = 1024 * 1024;  // 1,048,576
    constexpr int NUM_BLOCKS = NUM_FLOATS / BLOCK_SIZE;
    constexpr int BENCH_ITERS = 1000;

    std::printf("==========================================================\n");
    std::printf("  NibbleCore: 4-bit Quantization Benchmark (M4 ARM)\n");
    std::printf("==========================================================\n\n");

    std::printf("Configuration:\n");
    std::printf("  Floats:        %d (%.2f MB as float32)\n",
                NUM_FLOATS, NUM_FLOATS * 4.0 / (1024 * 1024));
    std::printf("  Blocks:        %d (block size = %d)\n", NUM_BLOCKS, BLOCK_SIZE);
    std::printf("  Quantized:     %.2f MB (%.1fx compression)\n",
                NUM_BLOCKS * sizeof(BlockQ4_0) / (1024.0 * 1024.0),
                (NUM_FLOATS * 4.0) / (NUM_BLOCKS * sizeof(BlockQ4_0)));
    std::printf("  Bench iters:   %d\n\n", BENCH_ITERS);

    // Allocate memory
    std::vector<float> input(NUM_FLOATS);
    std::vector<float> output_scalar(NUM_FLOATS);
    std::vector<float> output_neon(NUM_FLOATS);
    std::vector<BlockQ4_0> quantized(NUM_BLOCKS);

    // Fill input with random floats in [-1, 1].
    // Real model weights are roughly normally distributed around 0,
    // so uniform [-1, 1] is a reasonable test distribution.
    std::srand(42);  // fixed seed for reproducibility
    for (int i = 0; i < NUM_FLOATS; i++) {
        input[i] = 2.0f * (static_cast<float>(std::rand()) / RAND_MAX) - 1.0f;
    }

    // ---- BENCHMARKS ----
    std::printf("--- Quantization (float32 -> 4-bit) ---\n");

    double ms_quant_scalar = Benchmark::run("Scalar quantize", BENCH_ITERS, [&]() {
        quantize_row_scalar(input.data(), quantized.data(), NUM_FLOATS);
    });

    double ms_quant_neon = Benchmark::run("NEON quantize", BENCH_ITERS, [&]() {
        quantize_row_neon(input.data(), quantized.data(), NUM_FLOATS);
    });

    std::printf("\n--- Dequantization (4-bit -> float32) ---\n");

    double ms_dequant_scalar = Benchmark::run("Scalar dequantize", BENCH_ITERS, [&]() {
        dequantize_row_scalar(quantized.data(), output_scalar.data(), NUM_FLOATS);
    });

    double ms_dequant_neon = Benchmark::run("NEON dequantize", BENCH_ITERS, [&]() {
        dequantize_row_neon(quantized.data(), output_neon.data(), NUM_FLOATS);
    });

    // ---- THROUGHPUT ----
    std::printf("\n--- Throughput ---\n");
    size_t input_bytes = NUM_FLOATS * sizeof(float);
    Benchmark::print_throughput("Scalar quantize", input_bytes * BENCH_ITERS, ms_quant_scalar);
    Benchmark::print_throughput("NEON quantize",   input_bytes * BENCH_ITERS, ms_quant_neon);
    Benchmark::print_throughput("Scalar dequantize", input_bytes * BENCH_ITERS, ms_dequant_scalar);
    Benchmark::print_throughput("NEON dequantize",   input_bytes * BENCH_ITERS, ms_dequant_neon);

    // ---- SPEEDUP ----
    std::printf("\n--- NEON Speedup ---\n");
    std::printf("  Quantize:   %.2fx faster\n", ms_quant_scalar / ms_quant_neon);
    std::printf("  Dequantize: %.2fx faster\n", ms_dequant_scalar / ms_dequant_neon);

    // ---- QUALITY ----
    // Run once more to get the final output for quality measurement
    quantize_row_scalar(input.data(), quantized.data(), NUM_FLOATS);
    dequantize_row_scalar(quantized.data(), output_scalar.data(), NUM_FLOATS);

    quantize_row_neon(input.data(), quantized.data(), NUM_FLOATS);
    dequantize_row_neon(quantized.data(), output_neon.data(), NUM_FLOATS);

    auto q_scalar = measure_quality(input.data(), output_scalar.data(), NUM_FLOATS);
    auto q_neon   = measure_quality(input.data(), output_neon.data(), NUM_FLOATS);

    std::printf("\n--- Reconstruction Quality ---\n");
    std::printf("  Scalar:  MSE = %.8f  |  Max Error = %.6f\n", q_scalar.mse, q_scalar.max_error);
    std::printf("  NEON:    MSE = %.8f  |  Max Error = %.6f\n", q_neon.mse, q_neon.max_error);

    // Compression summary
    std::printf("\n--- Summary ---\n");
    std::printf("  Original size:   %zu bytes (%.2f MB)\n",
                input_bytes, input_bytes / (1024.0 * 1024.0));
    size_t quant_bytes = NUM_BLOCKS * sizeof(BlockQ4_0);
    std::printf("  Quantized size:  %zu bytes (%.2f MB)\n",
                quant_bytes, quant_bytes / (1024.0 * 1024.0));
    std::printf("  Compression:     %.1fx\n",
                static_cast<double>(input_bytes) / quant_bytes);
    std::printf("  Space saved:     %.1f%%\n",
                (1.0 - static_cast<double>(quant_bytes) / input_bytes) * 100.0);

    std::printf("\n==========================================================\n");
    std::printf("  Build complete.\n");
    std::printf("==========================================================\n");

    return 0;
}
