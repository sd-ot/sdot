#pragma once

#include <string>
#include <array>

namespace asimd {
namespace processing_units {
namespace features {

struct NvidiaGpuInfoFeature {
    using                            A2  = std::array<int,2>;
    using                            A3  = std::array<int,3>;

    static std::string               name() { return "NvidiaGpuInfoFeature"; }

    std::size_t                      num = 0;                    /**< num gpu board */

    #define NGIF( TYPE, NAME, INFO ) TYPE name;
    #include                         "NvidiaGpuInfoFeaturesDecl.h"
    #undef NGIF
};


} // namespace features
} // namespace processing_units
} // namespace asimd
