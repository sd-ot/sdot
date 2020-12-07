#include "../data/TypeFactoryRegister.h"
#include "BasicCudaAllocator.h"

namespace parex {

// TypeFactory
static TypeFactoryRegister _0( { "parex::BasicCudaAllocator" }, []( Type *res ) {
    res->compilation_environment.includes << "<parex/hardware/BasicCudaAllocator.h>";
} );

BasicCudaAllocator::BasicCudaAllocator( int num_gpu ) : mem( this, num_gpu ) {
}


} // namespace parex
