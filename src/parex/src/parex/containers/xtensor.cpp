#include "../data/TypeFactoryRegister.h"
#include "xtensor.h"

namespace parex {

// CompilationEnvironment for xt::xarray and xt::xtensor
namespace { static TypeFactoryRegister _0( { "xt::xarray", "xt::xtensor" }, CompilationEnvironment{
    .includes = { "<parex/containers/xtensor.h>" },
    // .libraries = { "xtensor", "xtensor::optimize", "xtensor::use_xsimd" },
    .cmake_packages = { "xtl", "xtensor" },
    .include_directories = { "ext/xtensor/install/include", "ext/xsimd/install/include", "src/asimd/src" },
} ); }

} // namespace parex
