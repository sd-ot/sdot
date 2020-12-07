#pragma once

#include "../containers/CpuAllocator.h"
#include "Memory.h"

namespace parex {
namespace hardware_information {

/**
*/
class CpuMemory : public Memory {
public:
    virtual void        write_to_stream( std::ostream &os ) const override;
    virtual std::string allocator_type () const override;
    virtual void*       allocator_data () const override;

    CpuAllocator*       allocator;
};

} // namespace hardware_information
} // namespace parex
