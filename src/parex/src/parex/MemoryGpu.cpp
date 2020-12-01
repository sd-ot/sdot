#include "CompilationEnvironment.h"
#include "MemoryGpu.h"

MemoryGpu::MemoryGpu( int num_gpu ) : num_gpu( num_gpu ) {
}

std::string MemoryGpu::allocator( CompilationEnvironment &compilation_environment, Type *type ) const {
    compilation_environment.includes << "<asimd/allocators/GpuAllocator.h>";
    return "asimd::GpuAllocator<" + type->cpp_name() + ">";
}

std::string MemoryGpu::name() const {
    return "GPU";
}

MemoryGpu *MemoryGpu::gpu( int num ) {
    static std::vector<MemoryGpu> res{ { 0 } };
    return &res[ num ];
}
