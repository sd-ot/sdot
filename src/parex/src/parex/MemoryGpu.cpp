#include "MemoryGpu.h"

MemoryGpu::MemoryGpu( int num_gpu ) : num_gpu( num_gpu ) {
}

std::string MemoryGpu::allocator( CompilationEnvironment &/*compilation_environment*/, Type *type ) const {
    return "xt:";
}

std::string MemoryGpu::name() const {
    return "GPU";
}
