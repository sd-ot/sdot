#include "../plugins/CompilationEnvironment.h"
#include "../plugins/Src.h"
#include "CudaMemory.h"

namespace parex {
namespace hardware_information {

void CudaMemory::write_to_stream( std::ostream &os ) const {
    os << "CudaMemory(num_gpu=" << allocator.num_gpu << ",amount=" << allocator.amount << ",used=" << allocator.used << ")";
}

std::string CudaMemory::allocator_type() const {
    return TypeInfo<CudaAllocator>::name();
}

void *CudaMemory::allocator_data() const {
    return &allocator;
}

} // namespace hardware_information
} // namespace parex
