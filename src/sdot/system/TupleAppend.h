#pragma once

#include <tuple>

/**
*/
template<class Tuple,class NewLast>
struct TupleAppend;

template<class ...TupleItems,class NewLast>
struct TupleAppend<std::tuple<TupleItems...>,NewLast> {
    using T = std::tuple<TupleItems...,NewLast>;
};
