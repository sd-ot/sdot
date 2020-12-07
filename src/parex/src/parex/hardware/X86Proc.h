#pragma once

#include "../utility/BumpPointerPool.h"
#include "ProcessorWithFeatureMap.h"
#include "Memory.h"
#include <vector>

namespace parex {

/**
*/
class X86Proc : public ProcessorWithFeatureMap {
public:
    static void         get_locals ( BumpPointerPool &pool, std::vector<Processor *> &processors, std::vector<Memory *> &memories );

    virtual std::size_t ptr_size   () const override;
    virtual std::string name       () const override;

    std::size_t         ptr_size_; ///<
};

} // namespace parex
