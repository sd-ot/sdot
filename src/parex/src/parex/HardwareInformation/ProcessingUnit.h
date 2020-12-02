#pragma once

#include <string>
#include <vector>

/**
*/
class ProcessingUnit {
public:
    virtual                 ~ProcessingUnit();

    std::string              name          () const;

    std::vector<std::string> features;     ///< AVX2, ...
    int                      ptr_size;     ///< in bits
    std::string              category;     ///< X86, NvidiaGpu, ...
};

