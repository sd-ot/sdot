#include "../plugins/CompilationEnvironment.h"
#include "../containers/CpuAllocator.h"
#include "../plugins/Src.h"
#include "CpuMemory.h"

namespace parex {

CpuMemory::CpuMemory() {
    amount = get_phys_pages() * sysconf( _SC_PAGESIZE );
}

void CpuMemory::write_to_stream( std::ostream &os ) const {
    os << "CpuMemory(amount=" << amount << ",used=" << amount << ")";
}

std::string CpuMemory::allocator_type() const {
    return "parex::CpuAllocator";
}

void *parex::hardware_information::CpuMemory::allocator_data() {
    return &CpuAllocator::local;
}

} // namespace parex
