#pragma once

#include "ProcessingUnit.h"
#include <vector>
#include <memory>

namespace asimd {
namespace hardware_information {

/**

*/
class HwGraph {
public:
    using PPU               = std::unique_ptr<ProcessingUnit>;
    using VPPU              = std::vector<PPU>;

    /**/  HwGraph           ();

    VPPU  processing_units; ///<

private:
    void  get_local_info    ();
    PPU   local_cpu         ();
};

} // namespace hardware_information
} // namespace asimd
