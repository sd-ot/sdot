#pragma once

#include "Memory.h"

namespace parex {
namespace hardware_information {

/**
*/
class CudaMemory : public Memory {
public:
    virtual void        write_to_stream( std::ostream &os ) const override;
    virtual std::string kernel_type    ( CompilationEnvironment &compilation_environment ) const override;

    std::uint64_t       amount;
};

} // namespace hardware_information
} // namespace parex
