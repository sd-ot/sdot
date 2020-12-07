#pragma once

#include "Memory.h"

namespace parex {
class BasicCudaAllocator;

/**
*/
class CudaMemory : public Memory {
public:
    /**/                CudaMemory         ( BasicCudaAllocator *default_allocator, int num_gpu );

    virtual void        write_to_stream    ( std::ostream &os ) const override;
    virtual std::string allocator_type     () const override;
    virtual void*       allocator_data     () override;

    BasicCudaAllocator* default_allocator; ///<
    int                 num_gpu;           ///<
};

} // namespace parex
