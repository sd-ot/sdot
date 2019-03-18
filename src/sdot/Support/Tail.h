#pragma once

#include <tuple>

/**
*/
template<class T>
struct Tail;

// specialization
template<class H,class... T>
struct Tail<std::tuple<H,T...>> {
    using type = std::tuple<T...>;
};
