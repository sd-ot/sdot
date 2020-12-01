#include "CompilationEnvironment.h"
#include "MemoryCpu.h"
#include "Src.h"

MemoryCpu memory_cpu;

std::string MemoryCpu::allocator( CompilationEnvironment &compilation_environment, Type *type ) const {
    compilation_environment.includes << "<asimd/allocators/AlignedAllocator.h>";
    return "asimd::AlignedAllocator<" + type->cpp_name() + ",64>";
}

std::string MemoryCpu::name() const {
    return "CPU";
}
