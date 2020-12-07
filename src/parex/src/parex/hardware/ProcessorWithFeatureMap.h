#pragma once

#include "Processor.h"
#include <memory>
#include <map>

namespace parex {

/**
*/
class ProcessorWithFeatureMap : public Processor {
public:
    struct                            Feature    { std::string name; };

    virtual void                      asimd_init ( std::ostream &os, const std::string &var_name, const std::string &sp ) const override;
    virtual std::string               asimd_name () const override;

    std::map<std::string,std::string> features;  ///<
};

} // namespace parex
