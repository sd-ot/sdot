#include "../data/TypeFactoryRegister.h"
#include "CpuAllocator.h"

namespace parex {

static TypeFactoryRegister _0( { "parex::CpuAllocator" }, []( Type *res ) {
    res->compilation_environment.includes << "<parex/containers/CpuAllocator.h>";
} );

CpuAllocator CpuAllocator::local;

} // namespace parex
