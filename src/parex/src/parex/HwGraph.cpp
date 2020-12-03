#include <cpu_features/cpu_features_macros.h>
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
    processing_units.push_back( local_cpu() );
}

std::unique_ptr<ProcessingUnit> HwGraph::local_cpu() {
    if ( std::unique_ptr<ProcessingUnit> res = X86::local() ) return res;
    TODO;
}

} // namespace parex
