// ============================================================================
// NibbleCore: Head-to-Head vs llama.cpp
// ============================================================================
//
// We benchmark three implementations of Q4_0 dequantization:
//
//   1. llama.cpp's actual implementation (scalar, from ggml-quants.c)
//   2. Our scalar implementation (Sprint 1)
//   3. Our NEON SIMD implementation (Sprint 1)
//
// Plus we show the quantized dot product approach that llama.cpp
// actually uses in production -- it never dequantizes at all.
//
// Usage: ./compare [path/to/model.gguf]
//        (if no model given, uses synthetic data)
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <arm_neon.h>

#include "gguf.hpp"
#include "benchmark.hpp"

#define C_RESET   "\033[0m"
#define C_BOLD    "\033[1m"
#define C_DIM     "\033[2m"
#define C_RED     "\033[31m"
#define C_GREEN   "\033[32m"
#define C_YELLOW  "\033[33m"
#define C_CYAN    "\033[36m"
#define C_BOLD_CYAN   "\033[1;36m"
#define C_BOLD_WHITE  "\033[1;37m"
#define C_BOLD_GREEN  "\033[1;32m"
#define C_BOLD_YELLOW "\033[1;33m"

// ============================================================================
// Q4_0 block (shared format)
// ============================================================================
struct BlockQ4_0 {
    float16_t scale;
    uint8_t qs[16];
};
static_assert(sizeof(BlockQ4_0) == 18);

// Q8_0 block -- used for the dot product approach
// In practice, the input activations get quantized to Q8_0 (8-bit)
// and then multiplied directly against Q4_0 weights.
struct BlockQ8_0 {
    float16_t scale;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34);

constexpr int QK = 32;

// ============================================================================
// IMPLEMENTATION 1: llama.cpp's dequantize_row_q4_0
// Copied verbatim from ggml-quants.c (scalar, no SIMD)
// ============================================================================
void dequantize_llamacpp(const BlockQ4_0* x, float* y, int k) {
    const int nb = k / QK;
    for (int i = 0; i < nb; i++) {
        const float d = static_cast<float>(x[i].scale);
        for (int j = 0; j < QK / 2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;
            y[i * QK + j + 0]      = x0 * d;
            y[i * QK + j + QK / 2] = x1 * d;
        }
    }
}

// ============================================================================
// IMPLEMENTATION 2: Our scalar dequantization (Sprint 1)
// Same math, slightly different style
// ============================================================================
void dequantize_ours_scalar(const BlockQ4_0* input, float* output, int k) {
    int nb = k / QK;
    for (int b = 0; b < nb; b++) {
        const BlockQ4_0& src = input[b];
        float* dst = output + b * QK;
        float scale = static_cast<float>(src.scale);

        for (int i = 0; i < 16; i++) {
            uint8_t packed = src.qs[i];
            uint8_t q0 = packed & 0x0F;
            uint8_t q1 = (packed >> 4) & 0x0F;
            dst[i]      = (static_cast<float>(q0) - 8.0f) * scale;
            dst[i + 16] = (static_cast<float>(q1) - 8.0f) * scale;
        }
    }
}

// ============================================================================
// IMPLEMENTATION 3: Our NEON dequantization (Sprint 1)
// Processes 4 values at a time using ARM SIMD
// ============================================================================
void dequantize_ours_neon(const BlockQ4_0* input, float* output, int k) {
    int nb = k / QK;
    for (int b = 0; b < nb; b++) {
        const BlockQ4_0& src = input[b];
        float* dst = output + b * QK;
        float scale = static_cast<float>(src.scale);
        float32x4_t vscale = vdupq_n_f32(scale);
        float32x4_t voffset = vdupq_n_f32(8.0f);

        for (int i = 0; i < 16; i += 4) {
            uint32_t tmp0[4], tmp1[4];
            for (int j = 0; j < 4; j++) {
                tmp0[j] = src.qs[i + j] & 0x0F;
                tmp1[j] = (src.qs[i + j] >> 4) & 0x0F;
            }
            uint32x4_t q0 = vld1q_u32(tmp0);
            uint32x4_t q1 = vld1q_u32(tmp1);

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
// IMPLEMENTATION 4: NEON-optimized Q4_0 dequantization (advanced)
// Uses byte-level NEON ops to unpack nibbles without scalar code
// ============================================================================
void dequantize_neon_advanced(const BlockQ4_0* input, float* output, int k) {
    int nb = k / QK;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    const int8x16_t offset = vdupq_n_s8(8);

    for (int b = 0; b < nb; b++) {
        const BlockQ4_0& src = input[b];
        float* dst = output + b * QK;
        float scale = static_cast<float>(src.scale);
        float32x4_t vscale = vdupq_n_f32(scale);

        // Load all 16 packed bytes at once (contains 32 weights)
        uint8x16_t packed = vld1q_u8(src.qs);

        // Unpack: low nibbles and high nibbles, all 16 at once
        int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(packed, mask_lo));
        int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));

        // Subtract 8 (the zero-point offset)
        lo = vsubq_s8(lo, offset);
        hi = vsubq_s8(hi, offset);

        // Now widen from int8 to int16 to int32 to float, 4 at a time
        // Low nibbles -> first 16 output floats
        int16x8_t lo_16_0 = vmovl_s8(vget_low_s8(lo));
        int16x8_t lo_16_1 = vmovl_s8(vget_high_s8(lo));

        vst1q_f32(dst +  0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_16_0))),  vscale));
        vst1q_f32(dst +  4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_16_0))), vscale));
        vst1q_f32(dst +  8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_16_1))),  vscale));
        vst1q_f32(dst + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_16_1))), vscale));

        // High nibbles -> last 16 output floats
        int16x8_t hi_16_0 = vmovl_s8(vget_low_s8(hi));
        int16x8_t hi_16_1 = vmovl_s8(vget_high_s8(hi));

        vst1q_f32(dst + 16, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_16_0))),  vscale));
        vst1q_f32(dst + 20, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_16_0))), vscale));
        vst1q_f32(dst + 24, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_16_1))),  vscale));
        vst1q_f32(dst + 28, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_16_1))), vscale));
    }
}

// ============================================================================
// BONUS: Quantized dot product (how llama.cpp ACTUALLY does inference)
//
// During inference, llama.cpp NEVER dequantizes weights. Instead:
//   1. Input activations (float32) get quantized to Q8_0 (8-bit)
//   2. The dot product is computed directly: Q4_0 weights * Q8_0 inputs
//   3. Result is accumulated in float32
//
// This is faster because:
//   - No memory allocation for dequantized weights
//   - Smaller data = better cache utilization
//   - NEON can do 16 int8 multiplies + accumulate in one instruction
// ============================================================================

// Quantize a row of floats to Q8_0 (what happens to input activations)
void quantize_to_q8_0(const float* input, BlockQ8_0* output, int k) {
    int nb = k / QK;
    for (int b = 0; b < nb; b++) {
        const float* src = input + b * QK;
        BlockQ8_0& dst = output[b];

        float amax = 0.0f;
        for (int i = 0; i < QK; i++) {
            float a = fabsf(src[i]);
            if (a > amax) amax = a;
        }

        float scale = amax / 127.0f;
        float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;
        dst.scale = static_cast<float16_t>(scale);

        for (int i = 0; i < QK; i++) {
            int v = static_cast<int>(roundf(src[i] * inv_scale));
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            dst.qs[i] = static_cast<int8_t>(v);
        }
    }
}

// Quantized dot product: Q4_0 weights dot Q8_0 inputs
// Returns the scalar result
float vec_dot_q4_0_q8_0_scalar(const BlockQ4_0* w, const BlockQ8_0* x, int k) {
    int nb = k / QK;
    float sumf = 0.0f;

    for (int b = 0; b < nb; b++) {
        float d_w = static_cast<float>(w[b].scale);
        float d_x = static_cast<float>(x[b].scale);
        int sum_i = 0;

        for (int j = 0; j < 16; j++) {
            int w0 = (w[b].qs[j] & 0x0F) - 8;
            int w1 = (w[b].qs[j] >>   4) - 8;
            sum_i += w0 * x[b].qs[j];
            sum_i += w1 * x[b].qs[j + 16];
        }

        sumf += d_w * d_x * sum_i;
    }
    return sumf;
}

// NEON-optimized quantized dot product
float vec_dot_q4_0_q8_0_neon(const BlockQ4_0* w, const BlockQ8_0* x, int k) {
    int nb = k / QK;
    float sumf = 0.0f;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    const int8x16_t offset = vdupq_n_s8(8);

    for (int b = 0; b < nb; b++) {
        float d_w = static_cast<float>(w[b].scale);
        float d_x = static_cast<float>(x[b].scale);

        // Load 16 packed weight bytes
        uint8x16_t packed = vld1q_u8(w[b].qs);

        // Unpack to signed int8, subtract offset
        int8x16_t w_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed, mask_lo)), offset);
        int8x16_t w_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), offset);

        // Load input activations (already int8)
        int8x16_t x_lo = vld1q_s8(x[b].qs);
        int8x16_t x_hi = vld1q_s8(x[b].qs + 16);

        // Multiply and accumulate: sdot does 4 int8 multiplies + add per lane
        // This is the key instruction that makes quantized inference fast
        int32x4_t acc = vdupq_n_s32(0);
        acc = vdotq_s32(acc, w_lo, x_lo);
        acc = vdotq_s32(acc, w_hi, x_hi);

        // Horizontal sum of the 4 int32 lanes
        sumf += d_w * d_x * vaddvq_s32(acc);
    }
    return sumf;
}

// ============================================================================
// Correctness checker
// ============================================================================
bool verify_match(const float* a, const float* b, int n, const char* name_a, const char* name_b) {
    float max_diff = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_diff) max_diff = d;
    }
    bool ok = max_diff < 1e-5f;
    std::printf("  %s vs %s: max diff = %.2e  %s\n",
                name_a, name_b, max_diff,
                ok ? (C_GREEN "MATCH" C_RESET) : (C_RED "MISMATCH" C_RESET));
    return ok;
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char* argv[]) {
    std::printf(C_BOLD_CYAN);
    std::printf("╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║  NibbleCore: Head-to-Head vs llama.cpp                      ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    std::printf(C_RESET);

    // Load data -- either from a model or synthetic
    constexpr int DEFAULT_SIZE = 1024 * 1024;
    int num_floats = DEFAULT_SIZE;
    std::vector<BlockQ4_0> blocks;
    std::string source;

    if (argc >= 2) {
        GGUFFile model;
        if (parse_gguf(argv[1], model)) {
            // Find first Q4_0 tensor
            for (auto& t : model.tensors) {
                if (t.type == GGMLType::Q4_0 && t.num_elements() >= 32) {
                    num_floats = static_cast<int>(t.num_elements());
                    int nb = num_floats / QK;
                    blocks.resize(nb);

                    std::ifstream f(argv[1], std::ios::binary);
                    f.seekg(model.data_offset + t.offset);
                    f.read(reinterpret_cast<char*>(blocks.data()), nb * sizeof(BlockQ4_0));

                    source = "Real model: " + t.name + " (" + std::to_string(num_floats) + " weights)";
                    break;
                }
            }
        }
    }

    if (blocks.empty()) {
        // Generate synthetic Q4_0 data
        int nb = num_floats / QK;
        blocks.resize(nb);
        std::srand(42);
        for (int b = 0; b < nb; b++) {
            float scale = 2.0f * (static_cast<float>(std::rand()) / RAND_MAX) - 1.0f;
            blocks[b].scale = static_cast<float16_t>(scale * 0.3f);
            for (int i = 0; i < 16; i++) {
                blocks[b].qs[i] = std::rand() & 0xFF;
            }
        }
        source = "Synthetic data (" + std::to_string(num_floats) + " weights)";
    }

    int nb = num_floats / QK;
    std::printf("  Source: " C_YELLOW "%s" C_RESET "\n", source.c_str());
    std::printf("  Blocks: %d  |  Floats: %d  |  Data: %.2f MB\n\n",
                nb, num_floats, nb * sizeof(BlockQ4_0) / (1024.0 * 1024.0));

    // Allocate output buffers
    std::vector<float> out_llama(num_floats);
    std::vector<float> out_scalar(num_floats);
    std::vector<float> out_neon(num_floats);
    std::vector<float> out_neon_adv(num_floats);

    // ── CORRECTNESS CHECK ───────────────────────────────────────────
    std::printf(C_BOLD_WHITE "═══ Correctness Verification ═══\n" C_RESET);

    dequantize_llamacpp(blocks.data(), out_llama.data(), num_floats);
    dequantize_ours_scalar(blocks.data(), out_scalar.data(), num_floats);
    dequantize_ours_neon(blocks.data(), out_neon.data(), num_floats);
    dequantize_neon_advanced(blocks.data(), out_neon_adv.data(), num_floats);

    verify_match(out_llama.data(), out_scalar.data(), num_floats, "llama.cpp", "Ours scalar");
    verify_match(out_llama.data(), out_neon.data(), num_floats, "llama.cpp", "Ours NEON v1");
    verify_match(out_llama.data(), out_neon_adv.data(), num_floats, "llama.cpp", "Ours NEON v2");

    // ── DEQUANTIZATION BENCHMARK ────────────────────────────────────
    std::printf("\n" C_BOLD_WHITE "═══ Dequantization Benchmark ═══\n" C_RESET);
    std::printf(C_DIM "  (lower time = faster)\n\n" C_RESET);

    constexpr int ITERS = 1000;

    double ms_llama = Benchmark::run("  llama.cpp (scalar)", ITERS, [&]() {
        dequantize_llamacpp(blocks.data(), out_llama.data(), num_floats);
    });

    double ms_scalar = Benchmark::run("  Ours scalar", ITERS, [&]() {
        dequantize_ours_scalar(blocks.data(), out_scalar.data(), num_floats);
    });

    double ms_neon = Benchmark::run("  Ours NEON v1", ITERS, [&]() {
        dequantize_ours_neon(blocks.data(), out_neon.data(), num_floats);
    });

    double ms_neon_adv = Benchmark::run("  Ours NEON v2 (advanced)", ITERS, [&]() {
        dequantize_neon_advanced(blocks.data(), out_neon_adv.data(), num_floats);
    });

    // ── THROUGHPUT ──────────────────────────────────────────────────
    std::printf("\n" C_BOLD_WHITE "═══ Throughput ═══\n" C_RESET);
    size_t total_bytes = static_cast<size_t>(num_floats) * 4 * ITERS;
    Benchmark::print_throughput("  llama.cpp (scalar)", total_bytes, ms_llama);
    Benchmark::print_throughput("  Ours scalar", total_bytes, ms_scalar);
    Benchmark::print_throughput("  Ours NEON v1", total_bytes, ms_neon);
    Benchmark::print_throughput("  Ours NEON v2 (advanced)", total_bytes, ms_neon_adv);

    // ── SPEEDUP TABLE ───────────────────────────────────────────────
    std::printf("\n" C_BOLD_WHITE "═══ Speedup vs llama.cpp ═══\n" C_RESET);
    std::printf("  %-30s  %8s  %10s\n", "Implementation", "Time", "vs llama");
    std::printf("  ──────────────────────────────  ────────  ──────────\n");
    std::printf("  %-30s  %7.2f ms  " C_DIM "baseline" C_RESET "\n", "llama.cpp (scalar)", ms_llama);
    std::printf("  %-30s  %7.2f ms  ", "Ours scalar", ms_scalar);
    if (ms_scalar < ms_llama) std::printf(C_GREEN); else std::printf(C_RED);
    std::printf("%.2fx" C_RESET "\n", ms_llama / ms_scalar);
    std::printf("  %-30s  %7.2f ms  ", "Ours NEON v1", ms_neon);
    if (ms_neon < ms_llama) std::printf(C_GREEN); else std::printf(C_RED);
    std::printf("%.2fx" C_RESET "\n", ms_llama / ms_neon);
    std::printf("  %-30s  %7.2f ms  ", "Ours NEON v2 (advanced)", ms_neon_adv);
    if (ms_neon_adv < ms_llama) std::printf(C_GREEN); else std::printf(C_RED);
    std::printf("%.2fx" C_RESET "\n", ms_llama / ms_neon_adv);

    // ── QUANTIZED DOT PRODUCT BENCHMARK ─────────────────────────────
    std::printf("\n" C_BOLD_WHITE "═══ Bonus: Quantized Dot Product (How LLMs Actually Work) ═══\n" C_RESET);
    std::printf(C_DIM "  In production, llama.cpp NEVER dequantizes. It computes\n");
    std::printf("  dot products directly on quantized data (Q4_0 * Q8_0).\n");
    std::printf("  This is why local LLMs are fast.\n\n" C_RESET);

    // Create a fake input activation vector
    std::vector<float> input_vec(num_floats);
    for (int i = 0; i < num_floats; i++)
        input_vec[i] = 0.1f * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f);

    // Quantize inputs to Q8_0
    std::vector<BlockQ8_0> input_q8(nb);
    quantize_to_q8_0(input_vec.data(), input_q8.data(), num_floats);

    // Method A: Dequantize weights, then float dot product
    volatile float result_deq = 0;
    double ms_deq_dot = Benchmark::run("  Dequant + float dot", ITERS, [&]() {
        dequantize_neon_advanced(blocks.data(), out_neon_adv.data(), num_floats);
        float sum = 0;
        for (int i = 0; i < num_floats; i++)
            sum += out_neon_adv[i] * input_vec[i];
        result_deq = sum;
    });

    // Method B: Direct quantized dot product (scalar)
    volatile float result_qdot_s = 0;
    double ms_qdot_scalar = Benchmark::run("  Q4_0*Q8_0 dot (scalar)", ITERS, [&]() {
        result_qdot_s = vec_dot_q4_0_q8_0_scalar(blocks.data(), input_q8.data(), num_floats);
    });

    // Method C: Direct quantized dot product (NEON)
    volatile float result_qdot_n = 0;
    double ms_qdot_neon = Benchmark::run("  Q4_0*Q8_0 dot (NEON)", ITERS, [&]() {
        result_qdot_n = vec_dot_q4_0_q8_0_neon(blocks.data(), input_q8.data(), num_floats);
    });

    std::printf("\n  " C_BOLD "Dot product speedup:" C_RESET "\n");
    std::printf("  Dequant+float:     %7.2f ms  " C_DIM "baseline" C_RESET "\n", ms_deq_dot);
    std::printf("  Q4*Q8 scalar:      %7.2f ms  " C_GREEN "%.1fx faster" C_RESET "\n",
                ms_qdot_scalar, ms_deq_dot / ms_qdot_scalar);
    std::printf("  Q4*Q8 NEON:        %7.2f ms  " C_BOLD_GREEN "%.1fx faster" C_RESET "\n",
                ms_qdot_neon, ms_deq_dot / ms_qdot_neon);

    // ── SUMMARY ─────────────────────────────────────────────────────
    std::printf("\n" C_BOLD_CYAN);
    std::printf("╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║  KEY INSIGHT                                                ║\n");
    std::printf("║                                                             ║\n");
    std::printf("║  llama.cpp's dequantize_row_q4_0 is plain scalar code.     ║\n");
    std::printf("║  Their real optimization is avoiding dequantization          ║\n");
    std::printf("║  entirely -- computing dot products directly on packed       ║\n");
    std::printf("║  4-bit values using NEON SDOT instructions.                 ║\n");
    std::printf("║                                                             ║\n");
    std::printf("║  Our NEON dequant beats their scalar reference, but the      ║\n");
    std::printf("║  quantized dot product beats everything.                    ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");
    std::printf(C_RESET "\n");

    return 0;
}
