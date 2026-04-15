// ============================================================================
// NibbleCore: GGUF Model Loader
// ============================================================================
//
// This is where it gets real. We open an actual model file, parse its
// structure, and dequantize real neural network weights using our
// Sprint 1 kernel.
//
// Usage: ./loader path/to/model.gguf
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <arm_neon.h>

#include "gguf.hpp"
#include "benchmark.hpp"

// ANSI colors
#define C_RESET   "\033[0m"
#define C_BOLD    "\033[1m"
#define C_DIM     "\033[2m"
#define C_RED     "\033[31m"
#define C_GREEN   "\033[32m"
#define C_YELLOW  "\033[33m"
#define C_BLUE    "\033[34m"
#define C_MAGENTA "\033[35m"
#define C_CYAN    "\033[36m"
#define C_BOLD_WHITE  "\033[1;37m"
#define C_BOLD_CYAN   "\033[1;36m"
#define C_BOLD_GREEN  "\033[1;32m"
#define C_BOLD_YELLOW "\033[1;33m"

// Our Q4_0 block -- identical to Sprint 1
struct BlockQ4_0 {
    float16_t scale;
    uint8_t qs[16];
};
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be 18 bytes");

// Dequantize a row of Q4_0 blocks into floats (from Sprint 1)
void dequantize_q4_0(const BlockQ4_0* blocks, float* output, int num_floats) {
    int num_blocks = num_floats / 32;
    for (int b = 0; b < num_blocks; b++) {
        const BlockQ4_0& src = blocks[b];
        float* dst = output + b * 32;
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

// Compute basic statistics on a float array
struct WeightStats {
    float min_val, max_val, mean, std_dev;
    float abs_mean;
    int num_zeros;      // weights that quantized to exactly 0
    int total;
};

WeightStats compute_stats(const float* data, int n) {
    WeightStats s{};
    s.total = n;
    s.min_val = data[0];
    s.max_val = data[0];

    double sum = 0, sum_sq = 0;
    s.num_zeros = 0;

    for (int i = 0; i < n; i++) {
        float v = data[i];
        if (v < s.min_val) s.min_val = v;
        if (v > s.max_val) s.max_val = v;
        sum += v;
        sum_sq += static_cast<double>(v) * v;
        if (v == 0.0f) s.num_zeros++;
    }

    s.mean = static_cast<float>(sum / n);
    s.std_dev = static_cast<float>(std::sqrt(sum_sq / n - s.mean * s.mean));
    s.abs_mean = static_cast<float>(std::abs(sum) / n);
    return s;
}

// Print a histogram of weight values
void print_histogram(const float* data, int n, int bins = 40) {
    float lo = data[0], hi = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] < lo) lo = data[i];
        if (data[i] > hi) hi = data[i];
    }

    std::vector<int> counts(bins, 0);
    float range = hi - lo;
    if (range == 0) range = 1;

    for (int i = 0; i < n; i++) {
        int bin = static_cast<int>((data[i] - lo) / range * (bins - 1));
        if (bin < 0) bin = 0;
        if (bin >= bins) bin = bins - 1;
        counts[bin]++;
    }

    int max_count = *std::max_element(counts.begin(), counts.end());

    std::printf("  " C_DIM "Distribution of dequantized weights:" C_RESET "\n");
    std::printf("  " C_DIM "%-8.4f" C_RESET, lo);
    int label_pad = bins - 16;
    for (int i = 0; i < label_pad; i++) std::printf(" ");
    std::printf(C_DIM "%8.4f\n" C_RESET, hi);

    for (int row = 8; row >= 0; row--) {
        std::printf("  ");
        for (int b = 0; b < bins; b++) {
            float threshold = max_count * (row / 8.0f);
            if (counts[b] > threshold) {
                // Color based on position: red for extremes, green for center
                float pos = static_cast<float>(b) / bins;
                if (pos < 0.2f || pos > 0.8f)
                    std::printf(C_RED "█" C_RESET);
                else if (pos < 0.35f || pos > 0.65f)
                    std::printf(C_YELLOW "█" C_RESET);
                else
                    std::printf(C_GREEN "█" C_RESET);
            } else {
                std::printf(" ");
            }
        }
        std::printf("\n");
    }
    std::printf("  ");
    for (int b = 0; b < bins; b++) std::printf(C_DIM "─" C_RESET);
    std::printf("\n");
}

// Format bytes into human-readable
std::string format_bytes(uint64_t bytes) {
    char buf[64];
    if (bytes >= 1024ULL * 1024 * 1024)
        std::snprintf(buf, sizeof(buf), "%.2f GB", bytes / (1024.0 * 1024 * 1024));
    else if (bytes >= 1024ULL * 1024)
        std::snprintf(buf, sizeof(buf), "%.2f MB", bytes / (1024.0 * 1024));
    else if (bytes >= 1024)
        std::snprintf(buf, sizeof(buf), "%.2f KB", bytes / 1024.0);
    else
        std::snprintf(buf, sizeof(buf), "%llu bytes", bytes);
    return buf;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::printf("Usage: %s <model.gguf>\n", argv[0]);
        std::printf("\nDownload a small model to test with:\n");
        std::printf("  curl -L -o models/smollm-135m-q4_0.gguf \\\n");
        std::printf("    'https://huggingface.co/leafspark/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_0.gguf'\n");
        return 1;
    }

    const char* model_path = argv[1];

    // ── PHASE 1: Parse the GGUF header ──────────────────────────────
    std::printf(C_BOLD C_CYAN);
    std::printf("╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║  NibbleCore: GGUF Model Loader                              ║\n");
    std::printf("║  Loading real neural network weights                        ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    std::printf(C_RESET);

    std::printf(C_BOLD "Loading: " C_YELLOW "%s" C_RESET "\n\n", model_path);

    GGUFFile model;
    if (!parse_gguf(model_path, model)) {
        return 1;
    }

    // ── PHASE 2: Display model info ─────────────────────────────────
    std::printf(C_BOLD C_WHITE "═══ Model Information ═══\n" C_RESET);
    std::printf("  GGUF version:    %u\n", model.version);
    std::printf("  Tensor count:    %llu\n", model.tensor_count);
    std::printf("  Metadata keys:   %zu\n", model.metadata.size());
    std::printf("  Data offset:     0x%llX (%s)\n\n",
                model.data_offset, format_bytes(model.data_offset).c_str());

    // Print interesting metadata
    std::printf(C_BOLD C_WHITE "═══ Metadata ═══\n" C_RESET);
    for (auto& [key, val] : model.metadata) {
        // Filter to interesting keys
        if (key.find("general.") == 0 ||
            key.find(".context_length") != std::string::npos ||
            key.find(".block_count") != std::string::npos ||
            key.find(".embedding_length") != std::string::npos ||
            key.find(".head_count") != std::string::npos ||
            key.find(".vocab_size") != std::string::npos ||
            key.find(".feed_forward") != std::string::npos) {

            std::printf("  " C_CYAN "%-40s" C_RESET " = ", key.c_str());
            switch (val.type) {
                case GGUFValueType::STRING:  std::printf(C_GREEN "\"%s\"" C_RESET, val.string_val.c_str()); break;
                case GGUFValueType::UINT32:  std::printf(C_YELLOW "%u" C_RESET, static_cast<uint32_t>(val.uint_val)); break;
                case GGUFValueType::UINT64:  std::printf(C_YELLOW "%llu" C_RESET, val.uint_val); break;
                case GGUFValueType::INT32:   std::printf(C_YELLOW "%d" C_RESET, static_cast<int32_t>(val.int_val)); break;
                case GGUFValueType::FLOAT32: std::printf(C_YELLOW "%.4f" C_RESET, val.float_val); break;
                case GGUFValueType::BOOL:    std::printf(C_YELLOW "%s" C_RESET, val.uint_val ? "true" : "false"); break;
                default: std::printf(C_DIM "(type %u)" C_RESET, static_cast<uint32_t>(val.type)); break;
            }
            std::printf("\n");
        }
    }

    // ── PHASE 3: Tensor overview ────────────────────────────────────
    std::printf("\n" C_BOLD C_WHITE "═══ Tensors ═══\n" C_RESET);

    uint64_t total_params = 0;
    uint64_t total_bytes = 0;
    uint64_t q4_0_count = 0;
    uint64_t q4_0_params = 0;

    // Count by type
    std::unordered_map<uint32_t, uint64_t> type_counts;
    std::unordered_map<uint32_t, uint64_t> type_params;

    for (auto& t : model.tensors) {
        uint64_t elems = t.num_elements();
        uint64_t bytes = t.data_size();
        total_params += elems;
        total_bytes += bytes;
        type_counts[static_cast<uint32_t>(t.type)]++;
        type_params[static_cast<uint32_t>(t.type)] += elems;

        if (t.type == GGMLType::Q4_0) {
            q4_0_count++;
            q4_0_params += elems;
        }
    }

    // Print type summary
    std::printf("  " C_BOLD "%-10s  %8s  %15s  %12s\n" C_RESET,
                "Type", "Tensors", "Parameters", "Size");
    std::printf("  ──────────  ────────  ───────────────  ────────────\n");

    for (auto& [type_id, count] : type_counts) {
        GGMLType t = static_cast<GGMLType>(type_id);
        uint64_t params = type_params[type_id];
        float bpw = ggml_type_bits_per_weight(t);
        uint64_t approx_bytes = static_cast<uint64_t>(params * bpw / 8.0f);
        std::printf("  " C_CYAN "%-10s" C_RESET "  %8llu  %15llu  %12s\n",
                    ggml_type_name(t), count, params, format_bytes(approx_bytes).c_str());
    }

    std::printf("  ──────────  ────────  ───────────────  ────────────\n");
    std::printf("  " C_BOLD "Total" C_RESET "       %8llu  %15llu  %12s\n",
                model.tensor_count, total_params, format_bytes(total_bytes).c_str());

    // What the model would be at full precision
    uint64_t fp32_bytes = total_params * 4;
    std::printf("\n  " C_BOLD "If stored as float32:" C_RESET " %s\n", format_bytes(fp32_bytes).c_str());
    std::printf("  " C_BOLD "Actual quantized:    " C_RESET " %s\n", format_bytes(total_bytes).c_str());
    std::printf("  " C_BOLD C_GREEN "Compression:          %.1fx" C_RESET "\n",
                static_cast<double>(fp32_bytes) / total_bytes);

    // ── PHASE 4: List some tensors ──────────────────────────────────
    std::printf("\n" C_BOLD C_WHITE "═══ Tensor List (first 20) ═══\n" C_RESET);
    std::printf("  " C_BOLD "%-45s  %-6s  %-20s  %12s\n" C_RESET,
                "Name", "Type", "Shape", "Size");
    std::printf("  ─────────────────────────────────────────────  ──────  ────────────────────  ────────────\n");

    int show_count = std::min(static_cast<int>(model.tensors.size()), 20);
    for (int i = 0; i < show_count; i++) {
        auto& t = model.tensors[i];
        std::string shape_str;
        for (uint32_t d = 0; d < t.n_dims; d++) {
            if (d > 0) shape_str += " x ";
            shape_str += std::to_string(t.dimensions[d]);
        }
        std::printf("  %-45s  " C_CYAN "%-6s" C_RESET "  %-20s  %12s\n",
                    t.name.c_str(),
                    ggml_type_name(t.type),
                    shape_str.c_str(),
                    format_bytes(t.data_size()).c_str());
    }
    if (model.tensors.size() > 20) {
        std::printf("  " C_DIM "... and %llu more tensors" C_RESET "\n",
                    model.tensor_count - 20);
    }

    // ── PHASE 5: Dequantize a real tensor ───────────────────────────
    // Find the first Q4_0 tensor to dequantize
    const GGUFTensorInfo* target = nullptr;
    for (auto& t : model.tensors) {
        if (t.type == GGMLType::Q4_0 && t.num_elements() > 0) {
            target = &t;
            break;
        }
    }

    if (!target) {
        std::printf("\n" C_YELLOW "No Q4_0 tensors found to dequantize.\n" C_RESET);
        std::printf("This model may use a different quantization format.\n");
        return 0;
    }

    std::printf("\n" C_BOLD C_WHITE "═══ Dequantizing Real Weights ═══\n" C_RESET);
    std::printf("  Target tensor: " C_YELLOW "%s" C_RESET "\n", target->name.c_str());
    std::printf("  Elements:      %llu\n", target->num_elements());
    std::printf("  Quantized:     %s\n", format_bytes(target->data_size()).c_str());
    std::printf("  Unquantized:   %s\n\n",
                format_bytes(target->num_elements() * 4).c_str());

    // Read the raw quantized data from the file
    uint64_t abs_offset = model.data_offset + target->offset;
    uint64_t data_size = target->data_size();
    uint64_t num_elements = target->num_elements();

    std::ifstream f(model_path, std::ios::binary);
    f.seekg(abs_offset);

    std::vector<uint8_t> raw_data(data_size);
    f.read(reinterpret_cast<char*>(raw_data.data()), data_size);
    f.close();

    if (!f.good() && !f.eof()) {
        std::fprintf(stderr, "Error reading tensor data\n");
        return 1;
    }

    // Dequantize using our Sprint 1 kernel
    const BlockQ4_0* blocks = reinterpret_cast<const BlockQ4_0*>(raw_data.data());
    std::vector<float> weights(num_elements);

    std::printf("  " C_BOLD "Dequantizing..." C_RESET);
    std::fflush(stdout);

    double ms = Benchmark::run("\n  Dequantize time", 1, [&]() {
        dequantize_q4_0(blocks, weights.data(), num_elements);
    });

    Benchmark::print_throughput("  Throughput", num_elements * 4, ms);

    // ── PHASE 6: Analyze the weights ────────────────────────────────
    std::printf("\n" C_BOLD C_WHITE "═══ Weight Analysis ═══\n" C_RESET);

    auto stats = compute_stats(weights.data(), num_elements);
    std::printf("  Min:      " C_RED    "%+.6f" C_RESET "\n", stats.min_val);
    std::printf("  Max:      " C_GREEN  "%+.6f" C_RESET "\n", stats.max_val);
    std::printf("  Mean:     " C_YELLOW "%+.6f" C_RESET "\n", stats.mean);
    std::printf("  Std Dev:  " C_YELLOW "%.6f" C_RESET "\n", stats.std_dev);
    std::printf("  Zeros:    %d / %d (%.1f%%)\n",
                stats.num_zeros, stats.total,
                100.0 * stats.num_zeros / stats.total);

    // Print histogram
    std::printf("\n");
    print_histogram(weights.data(), std::min(static_cast<int>(num_elements), 100000));

    // Show first 16 actual weights
    std::printf("\n  " C_BOLD "First 16 dequantized weights:" C_RESET "\n");
    for (int i = 0; i < 16 && i < static_cast<int>(num_elements); i++) {
        std::printf("    [%2d] " C_YELLOW "%+.6f" C_RESET, i, weights[i]);
        // Mini bar
        std::printf("  ");
        int bar_len = static_cast<int>(fabsf(weights[i]) / stats.std_dev * 5);
        if (bar_len > 20) bar_len = 20;
        const char* color = weights[i] >= 0 ? C_GREEN : C_RED;
        std::printf("%s", color);
        for (int b = 0; b < bar_len; b++) std::printf("█");
        std::printf(C_RESET "\n");
    }

    // Show the raw Q4_0 blocks
    std::printf("\n  " C_BOLD "First 3 raw Q4_0 blocks:" C_RESET "\n");
    int blocks_to_show = std::min(3, static_cast<int>(num_elements / 32));
    for (int b = 0; b < blocks_to_show; b++) {
        const BlockQ4_0& blk = blocks[b];
        float s = static_cast<float>(blk.scale);
        std::printf("    Block %d: scale=" C_CYAN "%.6f" C_RESET "  packed=[", b, s);
        for (int i = 0; i < 16; i++) {
            if (i > 0) std::printf(" ");
            std::printf(C_MAGENTA "%02X" C_RESET, blk.qs[i]);
        }
        std::printf("]\n");
    }

    // ── Summary ─────────────────────────────────────────────────────
    std::printf("\n" C_BOLD C_CYAN);
    std::printf("╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║  You just loaded and dequantized real neural network        ║\n");
    std::printf("║  weights from a production model file.                      ║\n");
    std::printf("║                                                             ║\n");
    std::printf("║  These are the same weights that generate text when you     ║\n");
    std::printf("║  chat with an LLM. Each one was trained on trillions of     ║\n");
    std::printf("║  tokens of human text.                                      ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");
    std::printf(C_RESET "\n");

    return 0;
}
