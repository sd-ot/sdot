#include "NvidiaGpu.h"
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

void NvidiaGpu::get_locals( std::vector<std::unique_ptr<ProcessingUnit>> &pus ) {
    using A2 = std::array<int,2>;
    using A3 = std::array<int,3>;

    #ifdef HAS_CUDA_HEADER
    int nDevices;
    cudaGetDeviceCount( &nDevices );
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

        //
        std::unique_ptr<NvidiaGpu> res = std::make_unique<NvidiaGpu>();
        res->features[ "NvidiaGpuInfoFeature" ] = io.str();
        res->ptr_size_ = 8 * sizeof( void * );

        pus.push_back( std::move( res ) );
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
