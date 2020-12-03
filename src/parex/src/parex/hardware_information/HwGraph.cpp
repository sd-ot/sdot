#include <cpu_features/cpu_features_macros.h>
#include "../TODO.h"
#include "HwGraph.h"
#include "X86.h"

namespace asimd {
namespace hardware_information {

HwGraph::HwGraph() {
    get_local_info();
}

void HwGraph::get_local_info() {
    processing_units.push_back( local_cpu() );
}

std::unique_ptr<ProcessingUnit> HwGraph::local_cpu() {
    if ( std::unique_ptr<ProcessingUnit> res = X86::local() ) return res;
    TODO;
}

} // namespace hardware_information
} // namespace asimd
