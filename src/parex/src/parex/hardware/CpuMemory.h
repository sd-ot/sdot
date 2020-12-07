#pragma once

#include "Memory.h"

namespace parex {
class BasicCpuAllocator;

/**
*/
class CpuMemory : public Memory {
public:
    /**/                CpuMemory      ( BasicCpuAllocator *default_allocator );

    virtual void        write_to_stream( std::ostream &os ) const override;
    virtual std::string allocator_type () const override;
    virtual void*       allocator_data () override;

    BasicCpuAllocator*  default_allocator;
};

} // namespace parex
