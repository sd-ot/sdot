#include "../TypeFactoryRegistrar.h"
#include "xtensor.h"

namespace { static TypeFactoryRegistrar _( "xt::xarray", CompilationEnvironment{
    .includes = { "<parex/containers/xtensor.h>" },
    .include_directories = { "ext/xtensor/install/include", "ext/xsimd/install/include" },
    .cmake_packages = { "xtl", "xtensor" },
    .cmake_libraries = { "xtensor", "xtensor::optimize", "xtensor::use_xsimd" }
} ); }
