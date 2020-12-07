#pragma once

#include "../containers/CudaAllocator.h"
#include "Memory.h"

namespace parex {

/**
*/
class CudaMemory : public Memory {
public:
    virtual void        write_to_stream( std::ostream &os ) const override;
    virtual std::string allocator_type () const override;
    virtual void*       allocator_data () override;

    CudaAllocator       allocator;
};

} // namespace parex
