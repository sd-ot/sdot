#include <cpu_features/cpu_features_macros.h>
#include "../utility/TODO.h"
#include "CpuMemory.h"
#include "CudaProc.h"
#include "HwGraph.h"
#include "X86Proc.h"

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

int HwGraph::nb_cuda_devices() const {
    int res = 0;
    for( const std::unique_ptr<ProcessingUnit> &pu : processing_units )
        res += pu->cuda_device();
    return res;
}

HwGraph::Mem *HwGraph::local_memory() const {
    for( const std::unique_ptr<Mem> &mem : memories )
        if ( mem->is_local )
            return mem.get();
    return nullptr;
}

void HwGraph::get_local_info() {
    CudaProc::get_locals( processing_units, memories );
    X86Proc ::get_locals( processing_units, memories );
}

HwGraph *hw_graph() {
    static HwGraph res;
    return &res;
}

} // namespace parex
