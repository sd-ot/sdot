#pragma once

#include <x86intrin.h>
#include "MachineArch.h"
#include "S.h"

template<class Arch>
std::uint64_t popcnt( std::uint64_t mask, S<Arch> ) {
    return _mm_popcnt_u64( mask );
}

template<class Arch>
std::uint32_t popcnt( std::uint32_t mask, S<Arch> ) {
    return _mm_popcnt_u32( mask );
}
