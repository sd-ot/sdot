#pragma once

#include "IndexOfTypeInTuple.h"
#include "TuplePrepend.h"


/**
  Input: a tuple type (used as a list of types).

  Output: a typedef `T` which is a tuple with types appearing only one time.
*/
template<class Tuple>
struct TupleWithUniqueTypes {
    template<int size,class NextResult>
    struct Cat {
        using C = typename std::tuple_element<size-1,Tuple>::type;
        using T = typename Cat<
            size-1,
            typename TuplePrepend<
                C,
                NextResult,
                IndexOfTypeInTuple<Tuple,C>::value == size - 1
            >::T
        >::T;
    };
    template<class NextResult>
    struct Cat<0,NextResult> {
        using T = NextResult;
    };

    using T = typename Cat<std::tuple_size<Tuple>::value,std::tuple<>>::T;
};
