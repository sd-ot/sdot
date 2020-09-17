#include <cuda_runtime.h>
#include <vector>

/**
 
*/
struct GpuInfo {
    struct Unit {
        int nb_cores;
        int L2_size;
        int mem;
    };

    // https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units
    GpuInfo() {
        int count = 0;
        cudaGetDeviceCount( &count );
        units.resize( count );

        for( int i = 0; i < count; ++i ) {
            cudaSetDevice( i );
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties( &deviceProp, i );

            units[ i ].nb_cores = deviceProp.multiProcessorCount; // ConvertSMVer2Cores( deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount ); // 
            units[ i ].L2_size = deviceProp.l2CacheSize;
            units[ i ].mem = deviceProp.totalGlobalMem;
        }
    }

    std::vector<Unit> units;
};
