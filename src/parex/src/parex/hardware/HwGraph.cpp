#include <cpu_features/cpu_features_macros.h>
#include "../utility/TODO.h"
#include "CpuMemory.h"
#include "NvidiaGpu.h"
#include "HwGraph.h"
#include "X86.h"

namespace parex {
using namespace hardware_information;

HwGraph::HwGraph() {
    get_local_info();
}

void HwGraph::write_to_stream( std::ostream &os ) const {
    for( const std::unique_ptr<ProcessingUnit> &pu : processing_units )
        os << *pu << "\n";
    for( const std::unique_ptr<Mem> &mem : memories )
        os << *mem << "\n";
}

HwGraph::Mem *HwGraph::local_memory() const {
    for( const std::unique_ptr<Mem> &mem : memories )
        if ( mem->local )
            return mem.get();
    return nullptr;
}

void HwGraph::get_local_info() {
    NvidiaGpu::get_locals( processing_units, memories );
    X86      ::get_locals( processing_units, memories );
}

HwGraph *hw_graph() {
    static HwGraph res;
    return &res;
}

} // namespace parex
