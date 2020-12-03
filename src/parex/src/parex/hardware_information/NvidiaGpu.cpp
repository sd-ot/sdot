#include "NvidiaGpu.h"
#include "GpuMemory.h"
#include <sstream>
#include "../S.h"

#if __has_include(<cuda_runtime.h>)
#define HAS_CUDA_HEADER 1
#endif

#ifdef HAS_CUDA_HEADER
#include <cuda_runtime.h>
//// nsmake lib_name cudart
#endif //  HAS_CUDA_HEADER

namespace parex {
namespace hardware_information {

namespace {

template<std::size_t n,class U> void get_value( std::ostream &os, S<std::array<int,n>>, U value ) {
    os << "{";
    for( std::size_t i = 0; i < n; ++i )
        os << ( i ? "," : "" ) << value[ i ];
    os << "}";
}

template<class T,class U> void get_value( std::ostream &os, T, U value ) {
    os << value;
}

}

void NvidiaGpu::get_locals( std::vector<std::unique_ptr<ProcessingUnit>> &pus, std::vector<std::unique_ptr<Memory>> &memories ) {
    using A2 = std::array<int,2>;
    using A3 = std::array<int,3>;

    #ifdef HAS_CUDA_HEADER
    int nDevices;
    if ( cudaGetDeviceCount( &nDevices ) )
        return;

    for( int i = 0; i < nDevices; i++ ) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties( &prop, i );

        // summary of global information
        std::ostringstream io;
        io << "{ .num = " << i;
        #define NGIF( TYPE, NAME, INFO ) get_value( io << ", ." << #NAME << " = ", S<TYPE>(), prop.NAME );
        #include <asimd/processing_units/NvidiaGpuInfoFeaturesDecl.h>
        #undef NGIF
        io << " }";

        // ProcessingUnit
        std::unique_ptr<NvidiaGpu> gpu = std::make_unique<NvidiaGpu>();
        gpu->features[ "NvidiaGpuInfoFeature" ] = io.str();
        gpu->ptr_size_ = 8 * sizeof( void * );

        // Memory
        std::unique_ptr<GpuMemory> mem = std::make_unique<GpuMemory>();
        mem->amount = prop.totalGlobalMem;
        mem->register_link( {
            .processing_unit = gpu.get(),
            .bandwidth = prop.memoryClockRate * 1000.0 * prop.memoryBusWidth / 8
        } );

        // register
        memories.push_back( std::move( mem ) );
        pus.push_back( std::move( gpu ) );
    }
    #endif //  HAS_CUDA_HEADER
}

std::size_t NvidiaGpu::ptr_size() const {
    return ptr_size_;
}

std::string NvidiaGpu::name() const {
    return "NvidiaGpu";
}

} // namespace hardware_information
} // namespace parex
