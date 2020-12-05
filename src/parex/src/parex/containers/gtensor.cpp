#include "../data/TypeFactoryRegistrar.h"
#include "gtensor.h"

namespace parex {

// CompilationEnvironment for xt::xarray and xt::xtensor
namespace { static TypeFactoryRegistrar _0( { "parex::gtensor" }, CompilationEnvironment{
    .includes = { "<parex/containers/gtensor.h>" },
} ); }

} // namespace parex
