#pragma once

#include <string>

namespace parex {
namespace hardware_information {

/**
*/
class ProcessingUnit {
public:
    virtual                 ~ProcessingUnit ();

    virtual void             write_to_stream( std::ostream &os ) const;
    virtual bool             cuda_device    () const;
    virtual void             asimd_init     ( std::ostream &os, const std::string &var_name, const std::string &sp = "" ) const = 0;
    virtual std::string      asimd_name     () const = 0;
    virtual std::size_t      ptr_size       () const = 0;
    virtual std::string      name           () const = 0;
};

} // namespace hardware_information
} // namespace parex
