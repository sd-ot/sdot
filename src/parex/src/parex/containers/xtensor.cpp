#include "../TypeFactoryRegistrar.h"
#include "xtensor.h"

namespace { static TypeFactoryRegistrar _( "xt::xarray", { "<parex/containers/xtensor.h>" } ); }
