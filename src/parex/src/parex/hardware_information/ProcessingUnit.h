#pragma once

#include <string>

namespace asimd {
namespace hardware_information {

/**
*/
class ProcessingUnit {
public:
    virtual                 ~ProcessingUnit ();

    virtual void             write_to_stream( std::ostream &os ) const;
    virtual std::string      asimd_name     () const = 0;
    virtual std::size_t      ptr_size       () const = 0;
};

} // namespace hardware_information
} // namespace asimd
