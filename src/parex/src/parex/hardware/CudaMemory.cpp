#include "../plugins/CompilationEnvironment.h"
#include "../plugins/Src.h"
#include "CudaMemory.h"

namespace parex {
namespace hardware_information {

void CudaMemory::write_to_stream( std::ostream &os ) const {
    os << "CudaMemory(amount=" << amount << ")";
}

std::string CudaMemory::kernel_type( CompilationEnvironment &compilation_environment ) const {
    compilation_environment.includes << "<parex/CudaMemory.h>";
    return "parex::CudaMemory";
}

} // namespace hardware_information
} // namespace parex
