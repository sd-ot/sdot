#pragma once

#include "../utility/BumpPointerPool.h"
#include "Processor.h"
#include "Memory.h"
#include <vector>
#include <memory>

namespace parex {

/**

*/
class HwGraph {
public:
    /**/                     HwGraph        ();

    virtual void             write_to_stream( std::ostream &os ) const;
    virtual int              nb_cuda_devices() const;

    std::vector<Processor *> processors;    ///<
    std::vector<Memory *>    memories;      ///<

private:
    void                     get_local_info ();
    BumpPointerPool          pool;          ///<
};

HwGraph *default_hw_graph();

} // namespace parex
