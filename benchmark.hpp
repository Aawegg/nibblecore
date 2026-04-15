#pragma once

// <chrono> gives us high-resolution timers.
// <string> and <cstdio> are for printing results.
#include <chrono>
#include <string>
#include <cstdio>
#include <functional>

class Benchmark {
public:
    // Runs a function `iterations` times and prints how long it took.
    //
    // Why a template? So we can pass ANY callable thing -- a lambda,
    // a function pointer, whatever -- without overhead. The compiler
    // inlines it, so there's zero cost from the wrapper itself.
    template<typename Func>
    static double run(const std::string& name, int iterations, Func fn) {
        // "steady_clock" is a monotonic clock -- it never jumps backward
        // (unlike wall-clock time which can adjust for daylight savings etc).
        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < iterations; i++) {
            fn();
        }

        auto end = std::chrono::steady_clock::now();

        // Calculate elapsed time in different units
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double us_per_iter = (ms * 1000.0) / iterations;

        std::printf("%-30s | %6d iters | %10.3f ms total | %8.3f us/iter\n",
                    name.c_str(), iterations, ms, us_per_iter);

        return ms;
    }

    // Measures throughput: how many gigabytes per second we're processing.
    // This is the number that matters for LLM inference -- if you can't
    // move data fast enough, the GPU/CPU sits idle waiting.
    static void print_throughput(const std::string& name, size_t bytes, double ms) {
        double seconds = ms / 1000.0;
        double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        double gbps = gb / seconds;
        std::printf("%-30s | %8.2f GB/s throughput\n", name.c_str(), gbps);
    }
};
