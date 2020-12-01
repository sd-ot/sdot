#pragma once

#include "position.h"
#include <memory>

namespace asimd {

template<class A>
struct AllocatorTrait;

// specialization for std::allocator
template<class T>
struct AllocatorTrait<std::allocator<T>> {
    enum {                          alignment = 1 };

    static position::Cpu<alignment> position  ( const std::allocator<T> & = {} ) { return {}; }
};

} // namespace asimd
