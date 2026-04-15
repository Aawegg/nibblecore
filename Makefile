CXX      = clang++
# -O3:          Maximum optimization (the compiler rearranges your code for speed)
# -std=c++17:   Modern C++ standard
# -march=armv8.4-a+fp16: Target your M4's ARM instruction set with float16 support
# -mcpu=apple-m4: Tell the compiler exactly which chip to optimize for
# -ffast-math:  Allow math optimizations that break IEEE 754 strictness
#               (fine for ML -- we're already approximating everything)
CXXFLAGS = -O3 -std=c++17 -march=armv8.4-a+fp16 -mcpu=apple-m4 -ffast-math
TARGET   = surgeon

all: $(TARGET) demo loader compare

$(TARGET): main.cpp benchmark.hpp
	$(CXX) $(CXXFLAGS) main.cpp -o $(TARGET)

demo: demo.cpp
	$(CXX) $(CXXFLAGS) demo.cpp -o demo

loader: loader.cpp gguf.hpp benchmark.hpp
	$(CXX) $(CXXFLAGS) loader.cpp -o loader

compare: compare.cpp gguf.hpp benchmark.hpp
	$(CXX) $(CXXFLAGS) -march=armv8.4-a+fp16+dotprod compare.cpp -o compare

run: $(TARGET)
	./$(TARGET)

watch: demo
	./demo

load: loader
	./loader models/*.gguf

versus: compare
	./compare models/*.gguf

clean:
	rm -f $(TARGET) demo loader compare

.PHONY: all run clean
