#pragma once

#include "ProcessingUnit.h"
#include <memory>
#include <map>

namespace parex {
namespace hardware_information {

/**
*/
class ProcessingUnitWithFeatureMap : public ProcessingUnit {
public:
    struct                            Feature    { std::string name; };

    virtual void                      asimd_init ( std::ostream &os, const std::string &var_name, const std::string &sp ) const override;
    virtual std::string               asimd_name () const override;

    std::map<std::string,std::string> features;  ///<
};

} // namespace hardware_information
} // namespace parex
