#pragma once

#include "ProcessingUnit.h"
#include "Memory.h"
#include <vector>
#include <memory>

namespace parex {

/**

*/
class HwGraph {
public:
    using        Proc              = ProcessingUnit;
    using        Mem               = Memory;
    using        VPProc            = std::vector<std::unique_ptr<Proc>>;
    using        VPMem             = std::vector<std::unique_ptr<Mem>>;

    /**/         HwGraph           ();

    virtual void write_to_stream   ( std::ostream &os ) const;
    virtual int  nb_cuda_devices   () const;
    virtual Mem* local_memory      () const;

    VPProc       processing_units; ///<
    VPMem        memories;         ///<

private:
    void         get_local_info    ();
};

HwGraph *hw_graph();

} // namespace parex
