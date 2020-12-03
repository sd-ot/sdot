#pragma once

#include "hardware_information/ProcessingUnit.h"
#include "hardware_information/Memory.h"
#include <vector>
#include <memory>

namespace parex {

/**

*/
class HwGraph {
public:
    using    PMemory           = std::unique_ptr<hardware_information::Memory>;
    using    PProc             = std::unique_ptr<hardware_information::ProcessingUnit>;
    using    VPMemory          = std::vector<PMemory>;
    using    VPProc            = std::vector<PProc>;

    /**/     HwGraph           ();

    void     write_to_stream   ( std::ostream &os ) const;

    VPProc   processing_units; ///<
    VPMemory memories; ///<

private:
    void     get_local_info    ();
};

} // namespace parex
