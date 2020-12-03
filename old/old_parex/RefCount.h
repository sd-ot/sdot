#pragma once

#include <cstdint>

/**
*/
struct RefCount {
    void                increment() const { ++value; }
    bool                decrement() const { return --value == 0; }

    mutable std::size_t value    = 0;
};
