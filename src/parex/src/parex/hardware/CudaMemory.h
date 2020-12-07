#pragma once

#include "../containers/CudaAllocator.h"
#include "Memory.h"

namespace parex {
namespace hardware_information {

/**
*/
class CudaMemory : public Memory {
public:
    virtual void          write_to_stream( std::ostream &os ) const override;
    virtual std::string   allocator_type () const override;
    virtual void*         allocator_data () const override;

    mutable CudaAllocator allocator;
};

} // namespace hardware_information
} // namespace parex
