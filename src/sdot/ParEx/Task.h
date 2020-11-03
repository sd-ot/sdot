#pragma once

#include <vector>

namespace parex {
class Kernel;

class Task {
public:
    /**/        Task();

    Kernel*     kernel;
    std::size_t cpt_use;
};

} // namespace parex
