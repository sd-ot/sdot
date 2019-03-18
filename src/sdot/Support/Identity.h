#pragma once

#include <memory>

struct Identity {
    template<typename U>
    constexpr auto operator()( U &&v ) const noexcept -> decltype( std::forward<U>( v ) ) {
        return std::forward<U>( v );
    }
};
