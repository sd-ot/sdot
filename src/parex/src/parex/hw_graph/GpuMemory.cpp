#include "../CompilationEnvironment.h"
#include "GpuMemory.h"
#include "../Src.h"

namespace parex {
namespace hardware_information {

void GpuMemory::write_to_stream( std::ostream &os ) const {
    os << "GpuMemory";
}

std::string GpuMemory::kernel_type( CompilationEnvironment &compilation_environment ) const {
    compilation_environment.includes << "<parex/GpuMemory.h>";
    return "parex::GpuMemory";
}

} // namespace hardware_information
} // namespace parex
