#include "../data/TypeFactoryRegister.h"
#include "CudaAllocator.h"

namespace parex {

// TypeFactory
static TypeFactoryRegister _0( { "parex::CudaAllocator" }, []( Type *res ) {
    res->compilation_environment.includes << "<parex/containers/CudaAllocator.h>";
} );

} // namespace parex
