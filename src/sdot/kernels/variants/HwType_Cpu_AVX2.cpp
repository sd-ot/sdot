#define ARCH_COND hw_info.has_AVX2()
#define ARCH_NAME Arch::AVX2
#define ARCH_SCORE 2

//// nsmake cpp_flag -march=haswell

#include "HwType_Cpu_.def"
