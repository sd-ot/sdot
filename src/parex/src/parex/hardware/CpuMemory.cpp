#include "../plugins/CompilationEnvironment.h"
#include "../containers/CpuAllocator.h"
#include "../plugins/Src.h"
#include "CpuMemory.h"

namespace parex {
namespace hardware_information {

void CpuMemory::write_to_stream( std::ostream &os ) const {
    os << "CpuMemory(amount=" << allocator->amount << ")";
}

std::string CpuMemory::allocator_type() const {
    return "parex::CpuAllocator";
}

void *parex::hardware_information::CpuMemory::allocator_data() const {
    return &CpuAllocator::local;
}

} // namespace hardware_information
} // namespace parex
