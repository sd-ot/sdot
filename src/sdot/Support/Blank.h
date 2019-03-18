#pragma once

#include <tuple>

/**
*/
struct Blank {
    enum { nbr = 0 };

    constexpr bool  operator==( Blank ) const { return true; }

    constexpr Blank operator&&( Blank ) const { return {}; }

    auto tuple_off() const { return std::tuple<>(); }
};
