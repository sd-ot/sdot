#pragma once

#include "hardware_information/ProcessingUnit.h"
#include <vector>
#include <memory>

namespace parex {

/**

*/
class HwGraph {
public:
    using PPU               = std::unique_ptr<hardware_information::ProcessingUnit>;
    using VPPU              = std::vector<PPU>;

    /**/  HwGraph           ();

    void  write_to_stream   ( std::ostream &os ) const;

    VPPU  processing_units; ///<

private:
    void  get_local_info    ();
};

} // namespace parex
