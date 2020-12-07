#include "../plugins/CompilationEnvironment.h"
#include "../plugins/Src.h"
#include "GpuMemory.h"

namespace parex {
namespace hardware_information {

void GpuMemory::write_to_stream( std::ostream &os ) const {
    os << "GpuMemory(amount=" << amount << ")";
}

std::string GpuMemory::kernel_type( CompilationEnvironment &compilation_environment ) const {
    compilation_environment.includes << "<parex/GpuMemory.h>";
    return "parex::GpuMemory";
}

} // namespace hardware_information
} // namespace parex
