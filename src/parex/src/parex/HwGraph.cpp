#include <cpu_features/cpu_features_macros.h>
#include "hardware_information/NvidiaGpu.h"
#include "hardware_information/X86.h"
#include "HwGraph.h"
#include "TODO.h"

namespace parex {
using namespace hardware_information;

HwGraph::HwGraph() {
    get_local_info();
}

void HwGraph::write_to_stream( std::ostream &os ) const {
    for( const std::unique_ptr<ProcessingUnit> &pu : processing_units )
        os << *pu << "\n";
}

void HwGraph::get_local_info() {
    NvidiaGpu::get_locals( processing_units );
    X86      ::get_locals( processing_units );
}

} // namespace parex