#pragma once
// ============================================================================
// GGUF Format Parser
// ============================================================================
//
// GGUF (GPT-Generated Unified Format) is the standard file format for
// storing quantized LLM weights. Created by the llama.cpp project.
//
// This parser reads the header, metadata, and tensor descriptors so we
// can locate and dequantize the actual weight data.
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <algorithm>

// The magic bytes at the start of every GGUF file
constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" as uint32 little-endian: G(47) G(47) U(55) F(46)

// GGUF metadata value types
enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// Quantization types -- these tell us how each tensor's data is stored
enum class GGMLType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,   // ← This is OUR format from Sprint 1
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
};

const char* ggml_type_name(GGMLType type) {
    switch (type) {
        case GGMLType::F32:     return "F32";
        case GGMLType::F16:     return "F16";
        case GGMLType::Q4_0:    return "Q4_0";
        case GGMLType::Q4_1:    return "Q4_1";
        case GGMLType::Q5_0:    return "Q5_0";
        case GGMLType::Q5_1:    return "Q5_1";
        case GGMLType::Q8_0:    return "Q8_0";
        case GGMLType::Q8_1:    return "Q8_1";
        case GGMLType::Q2_K:    return "Q2_K";
        case GGMLType::Q3_K:    return "Q3_K";
        case GGMLType::Q4_K:    return "Q4_K";
        case GGMLType::Q5_K:    return "Q5_K";
        case GGMLType::Q6_K:    return "Q6_K";
        case GGMLType::Q8_K:    return "Q8_K";
        case GGMLType::BF16:    return "BF16";
        default:                return "UNKNOWN";
    }
}

// How many bytes per block for each quantization type
// A "block" is the smallest unit of quantized data (e.g., 32 weights for Q4_0)
size_t ggml_type_block_size(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return 1;
        case GGMLType::F16:  return 1;
        case GGMLType::Q4_0: return 32;
        case GGMLType::Q4_1: return 32;
        case GGMLType::Q5_0: return 32;
        case GGMLType::Q5_1: return 32;
        case GGMLType::Q8_0: return 32;
        case GGMLType::Q8_1: return 32;
        case GGMLType::Q4_K: return 256;
        case GGMLType::Q5_K: return 256;
        case GGMLType::Q6_K: return 256;
        default: return 0;
    }
}

size_t ggml_type_bytes_per_block(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return 4;
        case GGMLType::F16:  return 2;
        case GGMLType::Q4_0: return 18;  // 2 (scale) + 16 (packed) = our BlockQ4_0!
        case GGMLType::Q4_1: return 20;
        case GGMLType::Q5_0: return 22;
        case GGMLType::Q5_1: return 24;
        case GGMLType::Q8_0: return 34;
        case GGMLType::Q8_1: return 40;
        case GGMLType::Q4_K: return 144;
        case GGMLType::Q5_K: return 176;
        case GGMLType::Q6_K: return 210;
        default: return 0;
    }
}

// Bits per weight for calculating compression
float ggml_type_bits_per_weight(GGMLType type) {
    size_t block_sz = ggml_type_block_size(type);
    size_t byte_sz = ggml_type_bytes_per_block(type);
    if (block_sz == 0) return 0;
    return (byte_sz * 8.0f) / block_sz;
}

// Describes one tensor (weight matrix) in the model
struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims;
    std::vector<uint64_t> dimensions;
    GGMLType type;
    uint64_t offset;         // offset within the data section

    uint64_t num_elements() const {
        uint64_t n = 1;
        for (auto d : dimensions) n *= d;
        return n;
    }

    uint64_t data_size() const {
        uint64_t elems = num_elements();
        size_t block_sz = ggml_type_block_size(type);
        if (block_sz == 0) return 0;
        uint64_t num_blocks = (elems + block_sz - 1) / block_sz;
        return num_blocks * ggml_type_bytes_per_block(type);
    }
};

// Metadata value (simplified -- stores as string for display)
struct GGUFMetaValue {
    GGUFValueType type;
    std::string string_val;
    uint64_t uint_val = 0;
    int64_t int_val = 0;
    double float_val = 0.0;
};

// The complete parsed GGUF file
struct GGUFFile {
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;

    std::vector<std::pair<std::string, GGUFMetaValue>> metadata;
    std::vector<GGUFTensorInfo> tensors;

    uint64_t data_offset;    // where the raw tensor data starts in the file
    std::string filepath;

    // Find a metadata value by key
    const GGUFMetaValue* find_meta(const std::string& key) const {
        for (auto& [k, v] : metadata) {
            if (k == key) return &v;
        }
        return nullptr;
    }

    std::string get_string(const std::string& key, const std::string& fallback = "") const {
        auto* v = find_meta(key);
        return (v && v->type == GGUFValueType::STRING) ? v->string_val : fallback;
    }

    uint64_t get_uint(const std::string& key, uint64_t fallback = 0) const {
        auto* v = find_meta(key);
        return v ? v->uint_val : fallback;
    }
};

// ============================================================================
// Binary reader helper -- reads values from a file sequentially
// ============================================================================
class BinaryReader {
    std::ifstream& f_;
public:
    BinaryReader(std::ifstream& f) : f_(f) {}

    template<typename T>
    T read() {
        T val;
        f_.read(reinterpret_cast<char*>(&val), sizeof(T));
        return val;
    }

    std::string read_string() {
        uint64_t len = read<uint64_t>();
        std::string s(len, '\0');
        f_.read(s.data(), len);
        return s;
    }

    void skip(size_t bytes) {
        f_.seekg(bytes, std::ios::cur);
    }

    uint64_t tell() {
        return f_.tellg();
    }

    void seek(uint64_t pos) {
        f_.seekg(pos);
    }

    bool good() { return f_.good(); }

    // Skip a metadata value (for types we don't fully parse)
    void skip_value(GGUFValueType type) {
        switch (type) {
            case GGUFValueType::UINT8:
            case GGUFValueType::INT8:
            case GGUFValueType::BOOL:    skip(1); break;
            case GGUFValueType::UINT16:
            case GGUFValueType::INT16:   skip(2); break;
            case GGUFValueType::UINT32:
            case GGUFValueType::INT32:
            case GGUFValueType::FLOAT32: skip(4); break;
            case GGUFValueType::UINT64:
            case GGUFValueType::INT64:
            case GGUFValueType::FLOAT64: skip(8); break;
            case GGUFValueType::STRING:  read_string(); break;
            case GGUFValueType::ARRAY: {
                auto arr_type = read<GGUFValueType>();
                uint64_t arr_len = read<uint64_t>();
                for (uint64_t i = 0; i < arr_len; i++) {
                    skip_value(arr_type);
                }
                break;
            }
        }
    }
};

// ============================================================================
// Parse a GGUF file
// ============================================================================

bool parse_gguf(const std::string& path, GGUFFile& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::fprintf(stderr, "Error: cannot open '%s'\n", path.c_str());
        return false;
    }

    BinaryReader reader(f);
    out.filepath = path;

    // Read and validate magic
    uint32_t magic = reader.read<uint32_t>();
    if (magic != GGUF_MAGIC) {
        std::fprintf(stderr, "Error: not a GGUF file (magic: 0x%08X, expected 0x%08X)\n",
                     magic, GGUF_MAGIC);
        return false;
    }

    out.version = reader.read<uint32_t>();
    out.tensor_count = reader.read<uint64_t>();
    out.metadata_count = reader.read<uint64_t>();

    if (out.version < 2 || out.version > 3) {
        std::fprintf(stderr, "Warning: GGUF version %u (expected 2 or 3)\n", out.version);
    }

    // Read metadata key-value pairs
    out.metadata.reserve(out.metadata_count);
    for (uint64_t i = 0; i < out.metadata_count; i++) {
        std::string key = reader.read_string();
        auto val_type = reader.read<GGUFValueType>();

        GGUFMetaValue val;
        val.type = val_type;

        switch (val_type) {
            case GGUFValueType::UINT8:   val.uint_val = reader.read<uint8_t>(); break;
            case GGUFValueType::INT8:    val.int_val = reader.read<int8_t>(); break;
            case GGUFValueType::UINT16:  val.uint_val = reader.read<uint16_t>(); break;
            case GGUFValueType::INT16:   val.int_val = reader.read<int16_t>(); break;
            case GGUFValueType::UINT32:  val.uint_val = reader.read<uint32_t>(); break;
            case GGUFValueType::INT32:   val.int_val = reader.read<int32_t>(); break;
            case GGUFValueType::UINT64:  val.uint_val = reader.read<uint64_t>(); break;
            case GGUFValueType::INT64:   val.int_val = reader.read<int64_t>(); break;
            case GGUFValueType::FLOAT32: val.float_val = reader.read<float>(); break;
            case GGUFValueType::FLOAT64: val.float_val = reader.read<double>(); break;
            case GGUFValueType::BOOL:    val.uint_val = reader.read<uint8_t>(); break;
            case GGUFValueType::STRING:  val.string_val = reader.read_string(); break;
            case GGUFValueType::ARRAY:
                // For arrays, just skip them (we don't need tokenizer data etc.)
                reader.skip_value(val_type);
                val.string_val = "[array]";
                continue;  // skip adding to metadata since we re-read
            default:
                std::fprintf(stderr, "Warning: unknown metadata type %u for key '%s'\n",
                             static_cast<uint32_t>(val_type), key.c_str());
                break;
        }

        out.metadata.emplace_back(key, val);
    }

    // Read tensor descriptors
    out.tensors.resize(out.tensor_count);
    for (uint64_t i = 0; i < out.tensor_count; i++) {
        auto& t = out.tensors[i];
        t.name = reader.read_string();
        t.n_dims = reader.read<uint32_t>();
        t.dimensions.resize(t.n_dims);
        for (uint32_t d = 0; d < t.n_dims; d++) {
            t.dimensions[d] = reader.read<uint64_t>();
        }
        t.type = reader.read<GGMLType>();
        t.offset = reader.read<uint64_t>();
    }

    // Data section starts after alignment padding (default alignment = 32)
    uint64_t pos = reader.tell();
    uint64_t alignment = 32;

    // Check if there's a custom alignment in metadata
    auto* align_meta = out.find_meta("general.alignment");
    if (align_meta) alignment = align_meta->uint_val;

    out.data_offset = ((pos + alignment - 1) / alignment) * alignment;

    return true;
}
