#include "../data/TypeFactoryRegister.h"
#include "BasicCpuAllocator.h"

namespace parex {

static TypeFactoryRegister _0( { "parex::BasicCpuAllocator" }, []( Type *res ) {
    res->compilation_environment.includes << "<parex/hardware/BasicCpuAllocator.h>";
} );

BasicCpuAllocator::BasicCpuAllocator() : mem( this ) {
}

} // namespace parex
