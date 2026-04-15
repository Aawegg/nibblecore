// ============================================================================
// NibbleCore: VISUAL DEMO -- Watch quantization happen in real time
// ============================================================================
// This is the "glass box" version. Same math as main.cpp, but it prints
// every single step so you can see what's happening inside.

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <unistd.h>    // usleep -- lets us slow things down for visibility
#include <arm_neon.h>

// ANSI color codes for terminal output
#define C_RESET   "\033[0m"
#define C_BOLD    "\033[1m"
#define C_DIM     "\033[2m"
#define C_RED     "\033[31m"
#define C_GREEN   "\033[32m"
#define C_YELLOW  "\033[33m"
#define C_BLUE    "\033[34m"
#define C_MAGENTA "\033[35m"
#define C_CYAN    "\033[36m"
#define C_WHITE   "\033[37m"
#define C_BG_GRAY "\033[48;5;236m"

constexpr int BLOCK_SIZE = 32;
constexpr int DELAY_US = 60000;  // 60ms between steps -- fast enough to feel
                                  // real, slow enough to read

struct BlockQ4_0 {
    float16_t scale;
    uint8_t qs[16];
};

// Print a binary representation of a byte, color-coding the two nibbles
void print_byte_binary(uint8_t val) {
    // High nibble in magenta
    std::printf(C_MAGENTA);
    for (int b = 7; b >= 4; b--) std::printf("%d", (val >> b) & 1);
    // Low nibble in cyan
    std::printf(C_CYAN);
    for (int b = 3; b >= 0; b--) std::printf("%d", (val >> b) & 1);
    std::printf(C_RESET);
}

// Print a progress bar
void print_bar(float value, float max_val, int width, const char* color) {
    int filled = static_cast<int>((fabsf(value) / max_val) * width);
    if (filled > width) filled = width;
    std::printf("%s", color);
    for (int i = 0; i < filled; i++) std::printf("█");
    std::printf(C_DIM);
    for (int i = filled; i < width; i++) std::printf("░");
    std::printf(C_RESET);
}

void clear_screen() {
    std::printf("\033[2J\033[H");
}

void pause_step() {
    usleep(DELAY_US);
}

int main() {
    clear_screen();

    std::printf(C_BOLD C_CYAN);
    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║     NibbleCore: Live Quantization Visualizer            ║\n");
    std::printf("║     Watch 32 floats become 18 bytes in real time       ║\n");
    std::printf("╚══════════════════════════════════════════════════════════╝\n\n");
    std::printf(C_RESET);

    // ── PHASE 1: Generate sample weights ────────────────────────────
    float weights[BLOCK_SIZE];
    std::srand(42);

    std::printf(C_BOLD C_WHITE "PHASE 1: Original Model Weights (32 x float32 = 128 bytes)\n" C_RESET);
    std::printf(C_DIM "These simulate actual neural network weights.\n\n" C_RESET);
    pause_step();

    for (int i = 0; i < BLOCK_SIZE; i++) {
        weights[i] = 2.0f * (static_cast<float>(std::rand()) / RAND_MAX) - 1.0f;
        // Vary the magnitude to make it interesting
        weights[i] *= (1.0f + static_cast<float>(i % 5) * 0.5f);
    }

    // Print all weights with visual bars
    float display_max = 0;
    for (int i = 0; i < BLOCK_SIZE; i++)
        if (fabsf(weights[i]) > display_max) display_max = fabsf(weights[i]);

    for (int i = 0; i < BLOCK_SIZE; i++) {
        std::printf("  w[%2d] = " C_YELLOW "%+8.5f" C_RESET "  ", i, weights[i]);
        if (weights[i] >= 0)
            print_bar(weights[i], display_max, 20, C_GREEN);
        else
            print_bar(weights[i], display_max, 20, C_RED);
        std::printf("\n");
        usleep(30000);
    }

    std::printf("\n  " C_DIM "Memory: 32 floats x 4 bytes = " C_BOLD C_RED "128 bytes" C_RESET "\n\n");
    usleep(500000);

    // ── PHASE 2: Find the scale ─────────────────────────────────────
    std::printf(C_BOLD C_WHITE "PHASE 2: Finding the Scale Factor\n" C_RESET);
    std::printf(C_DIM "Scan all 32 values, find the biggest absolute value.\n" C_RESET);
    std::printf(C_DIM "This becomes our \"ruler\" for the block.\n\n" C_RESET);
    pause_step();

    float amax = 0.0f;
    int amax_idx = 0;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float a = fabsf(weights[i]);
        if (a > amax) {
            amax = a;
            amax_idx = i;
            std::printf("  Scanning w[%2d] = %+8.5f  |abs| = %.5f  " C_GREEN C_BOLD "← NEW MAX" C_RESET "\n",
                        i, weights[i], a);
        } else {
            std::printf("  Scanning w[%2d] = %+8.5f  |abs| = %.5f\n", i, weights[i], a);
        }
        usleep(20000);
    }

    float scale = amax / 8.0f;
    float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

    std::printf("\n  " C_BOLD "Absolute max: " C_YELLOW "%.5f" C_RESET " (at w[%d])\n", amax, amax_idx);
    std::printf("  " C_BOLD "Scale = max / 8 = " C_CYAN "%.5f" C_RESET "\n", scale);
    std::printf("  " C_DIM "Why /8? Because 4 bits = 16 values (0-15), center is 8.\n" C_RESET);
    std::printf("  " C_DIM "So values map from [-max, +max] to [0, 15].\n\n" C_RESET);
    usleep(800000);

    // ── PHASE 3: Quantize and Pack ──────────────────────────────────
    std::printf(C_BOLD C_WHITE "PHASE 3: Quantize & Pack (the magic)\n" C_RESET);
    std::printf(C_DIM "For each value: normalize → shift to [0,15] → pack 2 per byte\n\n" C_RESET);
    pause_step();

    BlockQ4_0 block;
    block.scale = static_cast<float16_t>(scale);
    memset(block.qs, 0, 16);

    std::printf("  " C_BOLD "%-5s  %-10s %-10s %-6s %-6s  %-5s %-12s\n" C_RESET,
                "Pair", "w[i]", "w[i+16]", "q_lo", "q_hi", "Byte", "Binary");
    std::printf("  ───── ────────── ────────── ────── ────── ───── ────────────\n");

    for (int i = 0; i < 16; i++) {
        // Quantize value at position i (low nibble)
        float v0 = weights[i] * inv_scale + 8.0f;
        uint8_t q0 = static_cast<uint8_t>(roundf(v0));
        if (q0 > 15) q0 = 15;

        // Quantize value at position i+16 (high nibble)
        float v1 = weights[i + 16] * inv_scale + 8.0f;
        uint8_t q1 = static_cast<uint8_t>(roundf(v1));
        if (q1 > 15) q1 = 15;

        // Pack into one byte
        uint8_t packed = q0 | (q1 << 4);
        block.qs[i] = packed;

        std::printf("  [%2d,%2d] " C_YELLOW "%+8.5f" C_RESET " " C_YELLOW "%+8.5f" C_RESET
                    "  " C_CYAN "%2d" C_RESET "     " C_MAGENTA "%2d" C_RESET
                    "    " C_BOLD "0x%02X" C_RESET "  ",
                    i, i + 16, weights[i], weights[i + 16], q0, q1, packed);
        print_byte_binary(packed);

        // Show the packing operation
        std::printf("  " C_DIM "(%d | %d<<4)" C_RESET, q0, q1);
        std::printf("\n");
        usleep(100000);
    }

    size_t packed_size = sizeof(float16_t) + 16;
    std::printf("\n  " C_DIM "Memory: 2 (scale) + 16 (packed) = " C_BOLD C_GREEN "18 bytes" C_RESET "\n");
    std::printf("  " C_BOLD "Compression: 128 → 18 bytes = " C_GREEN "7.1x" C_RESET "\n\n");
    usleep(800000);

    // ── PHASE 4: Dequantize ─────────────────────────────────────────
    std::printf(C_BOLD C_WHITE "PHASE 4: Dequantize (reconstruct the floats)\n" C_RESET);
    std::printf(C_DIM "Reverse the process: unpack byte → subtract 8 → multiply by scale\n\n" C_RESET);
    pause_step();

    float reconstructed[BLOCK_SIZE];
    float dequant_scale = static_cast<float>(block.scale);

    std::printf("  " C_BOLD "%-5s  %-12s %-6s %-12s %-12s %-10s\n" C_RESET,
                "Byte", "Binary", "q_val", "Original", "Recovered", "Error");
    std::printf("  ───── ──────────── ────── ──────────── ──────────── ──────────\n");

    for (int i = 0; i < 16; i++) {
        uint8_t packed = block.qs[i];
        uint8_t q0 = packed & 0x0F;
        uint8_t q1 = (packed >> 4) & 0x0F;

        reconstructed[i]      = (static_cast<float>(q0) - 8.0f) * dequant_scale;
        reconstructed[i + 16] = (static_cast<float>(q1) - 8.0f) * dequant_scale;

        float err0 = fabsf(weights[i] - reconstructed[i]);
        float err1 = fabsf(weights[i + 16] - reconstructed[i + 16]);

        // Low nibble
        std::printf("  0x%02X   ", packed);
        print_byte_binary(packed);
        std::printf("  " C_CYAN "%2d" C_RESET "    "
                    C_YELLOW "%+8.5f" C_RESET "   "
                    C_GREEN "%+8.5f" C_RESET "   ",
                    q0, weights[i], reconstructed[i]);

        // Color error: green if tiny, yellow if moderate, red if large
        if (err0 < 0.05f)
            std::printf(C_GREEN "%.6f" C_RESET "\n", err0);
        else if (err0 < 0.1f)
            std::printf(C_YELLOW "%.6f" C_RESET "\n", err0);
        else
            std::printf(C_RED "%.6f" C_RESET "\n", err0);

        // High nibble
        std::printf("         ");
        print_byte_binary(packed);
        std::printf("  " C_MAGENTA "%2d" C_RESET "    "
                    C_YELLOW "%+8.5f" C_RESET "   "
                    C_GREEN "%+8.5f" C_RESET "   ",
                    q1, weights[i + 16], reconstructed[i + 16]);

        if (err1 < 0.05f)
            std::printf(C_GREEN "%.6f" C_RESET "\n", err1);
        else if (err1 < 0.1f)
            std::printf(C_YELLOW "%.6f" C_RESET "\n", err1);
        else
            std::printf(C_RED "%.6f" C_RESET "\n", err1);

        usleep(100000);
    }

    // ── PHASE 5: Quality Summary ────────────────────────────────────
    std::printf("\n");
    usleep(500000);

    std::printf(C_BOLD C_WHITE "PHASE 5: Quality Report\n" C_RESET);
    std::printf(C_DIM "How close are the reconstructed values to the originals?\n\n" C_RESET);
    pause_step();

    double sum_sq = 0.0;
    float max_err = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float e = fabsf(weights[i] - reconstructed[i]);
        sum_sq += static_cast<double>(e) * e;
        if (e > max_err) max_err = e;
    }
    double mse = sum_sq / BLOCK_SIZE;

    std::printf("  ┌─────────────────────────────────────────────────┐\n");
    std::printf("  │  " C_BOLD "Original" C_RESET ":       32 x float32 = " C_RED "128 bytes" C_RESET "        │\n");
    std::printf("  │  " C_BOLD "Quantized" C_RESET ":      1 block Q4_0 = " C_GREEN " 18 bytes" C_RESET "        │\n");
    std::printf("  │  " C_BOLD "Compression" C_RESET ":    " C_BOLD C_GREEN "7.1x" C_RESET " (85.9%% saved)             │\n");
    std::printf("  │                                                 │\n");
    std::printf("  │  " C_BOLD "Mean Sq Error" C_RESET ":  " C_YELLOW "%.8f" C_RESET "                  │\n", mse);
    std::printf("  │  " C_BOLD "Max Error" C_RESET ":      " C_YELLOW "%.6f" C_RESET "                    │\n", max_err);
    std::printf("  │                                                 │\n");

    // Visual comparison
    std::printf("  │  " C_BOLD "Before" C_RESET ": ");
    for (int i = 0; i < 20; i++) {
        if (weights[i] >= 0) std::printf(C_GREEN "▓" C_RESET);
        else std::printf(C_RED "▓" C_RESET);
    }
    std::printf("  (32-bit)       │\n");

    std::printf("  │  " C_BOLD "After " C_RESET ": ");
    for (int i = 0; i < 20; i++) {
        if (reconstructed[i] >= 0) std::printf(C_GREEN "▒" C_RESET);
        else std::printf(C_RED "▒" C_RESET);
    }
    std::printf("  (4-bit)        │\n");

    std::printf("  └─────────────────────────────────────────────────┘\n\n");

    // Scale it up
    std::printf(C_BOLD C_CYAN "  AT SCALE (7B parameter model):\n" C_RESET);
    double model_params = 7e9;
    double original_gb = (model_params * 4.0) / (1024.0 * 1024.0 * 1024.0);
    double quantized_gb = (model_params / BLOCK_SIZE * 18.0) / (1024.0 * 1024.0 * 1024.0);
    std::printf("  Original:    " C_RED  "%.1f GB" C_RESET "\n", original_gb);
    std::printf("  Quantized:   " C_GREEN "%.1f GB" C_RESET "\n", quantized_gb);
    std::printf("  Your M4 Air: " C_CYAN "Fits." C_RESET "\n\n");

    std::printf(C_BOLD C_CYAN);
    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║  That's it. That's what llama.cpp does under the hood.  ║\n");
    std::printf("║  You just watched a neural network get compressed.      ║\n");
    std::printf("╚══════════════════════════════════════════════════════════╝\n");
    std::printf(C_RESET "\n");

    return 0;
}
