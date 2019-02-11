#pragma once

#include <tuple>

/**
*/
template<class Tuple,class Type,int off=0>
struct IndexOfTypeInTuple {
    enum { value = -1 };
};

template<class Head,class... Tail,class Type,int off>
struct IndexOfTypeInTuple<std::tuple<Head,Tail...>,Type,off> {
    enum { value = std::is_same<Type,Head>::value ? off : IndexOfTypeInTuple<std::tuple<Tail...>,Type,off+1>::value };
};
