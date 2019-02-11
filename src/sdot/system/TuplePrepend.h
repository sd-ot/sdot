#pragma once

#include <tuple>

/**
*/
template<class NewHead,class Tuple,bool condition=true>
struct TuplePrepend;

template<class NewHead,class ...TupleItems>
struct TuplePrepend<NewHead,std::tuple<TupleItems...>,true> {
    using T = std::tuple<NewHead,TupleItems...>;
};

template<class NewHead,class ...TupleItems>
struct TuplePrepend<NewHead,std::tuple<TupleItems...>,false> {
    using T = std::tuple<TupleItems...>;
};
