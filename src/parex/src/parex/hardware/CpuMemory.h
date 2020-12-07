#pragma once

#include "../containers/CpuAllocator.h"
#include "Memory.h"

namespace parex {

/**
*/
class CpuMemory : public Memory {
public:
    /**/                CpuMemory      ();

    virtual void        write_to_stream( std::ostream &os ) const override;
    virtual std::string allocator_type () const override;
    virtual void*       allocator_data () override;
};

} // namespace parex
