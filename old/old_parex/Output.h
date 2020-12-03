#pragma once

#include "Variant.h"
#include "RcPtr.h"
#include <vector>

/**
*/
class Output {
public:
    RefCount                    ref_count;
    std::vector<RcPtr<Variant>> variants;
};

