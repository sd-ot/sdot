#pragma once

#include "ProcessingUnit.h"
#include <memory>
#include <map>

namespace parex {
namespace hardware_information {

/**
*/
class X86 : public ProcessingUnit {
public:
    struct                            Feature    { std::string name; };

    virtual void                      asimd_init ( std::ostream &os, const std::string &var_name, const std::string &sp ) const override;
    virtual std::string               asimd_name () const override;
    virtual std::size_t               ptr_size   () const override;

    static std::unique_ptr<X86>       local      ();

    std::size_t                       ptr_size_; ///<
    std::map<std::string,std::string> features;  ///<
};

} // namespace hardware_information
} // namespace parex
