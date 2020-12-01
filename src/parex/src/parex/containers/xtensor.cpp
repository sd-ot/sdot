#include "../TypeFactoryRegistrar.h"
#include "xtensor.h"

// CompilationEnvironment for xt::xarray and xt::xtensor
namespace { static TypeFactoryRegistrar _0( { "xt::xarray", "xt::xtensor" }, CompilationEnvironment{
    .includes = { "<parex/containers/xtensor.h>" },
    // .libraries = { "xtensor", "xtensor::optimize", "xtensor::use_xsimd" },
    .cmake_packages = { "xtl", "xtensor" },
    .include_directories = { "ext/xtensor/install/include", "ext/xsimd/install/include", "src/asimd/src" },
} ); }
