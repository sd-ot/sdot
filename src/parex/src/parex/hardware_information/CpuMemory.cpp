#include "../CompilationEnvironment.h"
#include "CpuMemory.h"
#include "../Src.h"

namespace parex {
namespace hardware_information {

void CpuMemory::write_to_stream( std::ostream &os ) const {
    os << "CpuMemory(amount=" << amount << ")";
}

std::string CpuMemory::kernel_type( CompilationEnvironment &compilation_environment ) const {
    compilation_environment.includes << "<parex/CpuMemory.h>";
    return "parex::CpuMemory";
}

} // namespace hardware_information
} // namespace parex
