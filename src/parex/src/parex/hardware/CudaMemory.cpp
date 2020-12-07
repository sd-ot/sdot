#include "../plugins/CompilationEnvironment.h"
#include "../plugins/Src.h"
#include "CudaMemory.h"

namespace parex {

void CudaMemory::write_to_stream( std::ostream &os ) const {
    os << "CpuMemory(amount=" << amount << ",used=" << amount << ")";
}

std::string CudaMemory::allocator_type() const {
    return TypeInfo<CudaAllocator>::name();
}

void *CudaMemory::allocator_data() {
    return &allocator;
}

} // namespace parex
