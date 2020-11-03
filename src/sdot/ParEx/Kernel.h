#pragma once

#include <string>
#include <vector>

namespace parex {

/**
*/
class Kernel {
public:
    // where it can be executed
    struct Exe {
        bool cpu;
        bool gpu;
    };

    struct Inp {

    };

    struct Out {

    };

    std::vector<Out> outputs;
    std::vector<Inp> inputs;
};

} // namespace parex
