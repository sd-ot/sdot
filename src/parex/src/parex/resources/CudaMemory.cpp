#include "../plugins/CompilationEnvironment.h"
#include "../plugins/Src.h"
#include "BasicCudaAllocator.h"
#include "CudaMemory.h"

namespace parex {

CudaMemory::CudaMemory( BasicCudaAllocator *default_allocator, int num_gpu ) : default_allocator( default_allocator ), num_gpu( num_gpu ) {
}

void CudaMemory::write_to_stream( std::ostream &os ) const {
    os << "CudaMemory(amount=" << amount << ",used=" << used << ")";
}

std::string CudaMemory::allocator_type() const {
    return TypeInfo<BasicCudaAllocator>::name();
}

void *CudaMemory::allocator_data() {
    return &default_allocator;
}

} // namespace parex
