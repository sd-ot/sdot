#include "../utility/S.h"
#include "BasicCudaAllocator.h"
#include "CudaMemory.h"
#include "CudaProc.h"
#include <sstream>

#if __has_include(<cuda_runtime.h>)
#define HAS_CUDA_HEADER 1
#endif

#ifdef HAS_CUDA_HEADER
#include <cuda_runtime.h>
//// nsmake lib_name cudart
#endif //  HAS_CUDA_HEADER

namespace parex {

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

void CudaProc::get_locals( BumpPointerPool &pool, std::vector<Processor *> &processors, std::vector<Memory *> &memories ) {
    using A2 = std::array<int,2>;
    using A3 = std::array<int,3>;

    #ifdef HAS_CUDA_HEADER
    int nDevices;
    if ( cudaGetDeviceCount( &nDevices ) )
        return;

    for( int num_gpu = 0; num_gpu < nDevices; num_gpu++ ) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties( &prop, num_gpu );

        // summary of global information
        std::ostringstream io;
        io << "{ .num = " << num_gpu;
        #define NGIF( TYPE, NAME, INFO ) get_value( io << ", ." << #NAME << " = ", S<TYPE>(), prop.NAME );
        #include <asimd/processing_units/CudaProcInfoFeaturesDecl.h>
        #undef NGIF
        io << " }";

        // ProcessingUnit
        CudaProc *gpu = pool.create<CudaProc>();
        gpu->features[ "CudaProcInfoFeature" ] = io.str();
        gpu->ptr_size_ = 8 * sizeof( void * );

        // Memory
        BasicCudaAllocator *allocator = pool.create<BasicCudaAllocator>( num_gpu );
        allocator->mem.amount = prop.totalGlobalMem;

        allocator->mem.register_link( {
            .processing_unit = gpu,
            .bandwidth = prop.memoryClockRate * 1000.0 * prop.memoryBusWidth / 8
        } );

        // register
        memories.push_back( &allocator->mem );
        processors.push_back( gpu );
    }
    #endif //  HAS_CUDA_HEADER
}

bool CudaProc::cuda_device() const {
    return true;
}

std::size_t CudaProc::ptr_size() const {
    return ptr_size_;
}

std::string CudaProc::name() const {
    return "CudaProc";
}

} // namespace parex
