#include <cpu_features/cpu_features_macros.h>
#include "../utility/generic_ostream_output.h"
#include "default_CpuAllocator.h"
#include "CpuMemory.h"
#include "CudaProc.h"
#include "HwGraph.h"
#include "X86Proc.h"

namespace parex {

HwGraph::HwGraph() {
    get_local_info();
}

void HwGraph::write_to_stream( std::ostream &os ) const {
    for( const Processor *processor : processors )
        os << *processor << "\n";
    for( const Memory *memory : memories )
        os << *memory << "\n";
}

int HwGraph::nb_cuda_devices() const {
    int res = 0;
    for( const Processor *processor : processors )
        res += processor->cuda_device();
    return res;
}

void HwGraph::get_local_info() {
    // main memory
    memories.push_back( &default_CpuAllocator.mem );

    // processors with associated specific memories
    CudaProc::get_locals( pool, processors, memories );
    X86Proc ::get_locals( pool, processors, memories );
}

HwGraph *default_hw_graph() {
    static HwGraph res;
    return &res;
}

} // namespace parex
