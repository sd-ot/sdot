#pragma once

#include "ProcessingUnitWithFeatureMap.h"
#include <vector>

namespace parex {
namespace hardware_information {

/**
*/
class NvidiaGpu : public ProcessingUnitWithFeatureMap {
public:
    static void         get_locals ( std::vector<std::unique_ptr<ProcessingUnit>> &pus );
    virtual std::size_t ptr_size   () const override;
    virtual std::string name       () const override;

    std::size_t         ptr_size_; ///<
};

} // namespace hardware_information
} // namespace parex
