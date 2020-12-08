#include "../plugins/CompilationEnvironment.h"
#include "../plugins/Src.h"
#include "BasicCpuAllocator.h"
#include "CpuMemory.h"

#include <sys/sysinfo.h>
#include <unistd.h>

namespace parex {

CpuMemory::CpuMemory( BasicCpuAllocator *default_allocator ) : default_allocator( default_allocator ) {
    amount = get_phys_pages() * sysconf( _SC_PAGESIZE );
}

void CpuMemory::write_to_stream( std::ostream &os ) const {
    os << "CpuMemory(amount=" << amount << ",used=" << used << ")";
}

std::string CpuMemory::allocator_type() const {
    return "parex::BasicCpuAllocator";
}

void *CpuMemory::allocator_data() {
    return default_allocator;
}

} // namespace parex
