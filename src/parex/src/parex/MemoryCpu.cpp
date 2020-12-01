#include "CompilationEnvironment.h"
#include "MemoryCpu.h"
#include "Src.h"

MemoryCpu memory_cpu;

std::string MemoryCpu::allocator( CompilationEnvironment &compilation_environment, Type *type ) const {
    compilation_environment.includes << "<parex/containers/xtensor.h>";
    return "xsimd::aligned_allocator<" + type->cpp_name() + ",XSIMD_DEFAULT_ALIGNMENT>";
}

std::string MemoryCpu::name() const {
    return "CPU";
}
