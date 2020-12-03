#pragma once

#include "ProcessingUnit.h"
#include <memory>
#include <map>

namespace asimd {
namespace hardware_information {

/**
*/
class X86 : public ProcessingUnit {
public:
    struct                            Feature   { std::string name; };

    virtual std::string               asimd_name() const override;
    virtual std::size_t               ptr_size  () const override;

    static std::unique_ptr<X86>       local     ();

    std::map<std::string,std::string> features; ///>
};

} // namespace hardware_information
} // namespace asimd
