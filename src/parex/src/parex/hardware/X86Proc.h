#pragma once

#include "ProcessingUnitWithFeatureMap.h"
#include "Memory.h"
#include <vector>

namespace parex {

/**
*/
class X86Proc : public ProcessingUnitWithFeatureMap {
public:
    static void         get_locals ( std::vector<std::unique_ptr<ProcessingUnit>> &pus, std::vector<std::unique_ptr<Memory>> &memories );

    virtual std::size_t ptr_size   () const override;
    virtual std::string name       () const override;

    std::size_t         ptr_size_; ///<
};

} // namespace parex
