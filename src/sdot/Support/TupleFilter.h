#pragma once

#include "TuplePrepend.h"

/**
  Allow to construct a tuple from a a subset of another tuple
  Example of Filter:
      struct MyFilter {
          template<class U>
          struct Filter {
              enum { keep = sizeof( U ) == 1 };
          };
      };

*/
template<class Tuple,class Filter>
struct TupleFilter {
    using T = std::tuple<>;
};

/**
*/
template<class Head,class... Tail,class Filter>
struct TupleFilter<std::tuple<Head,Tail...>,Filter> {
    template<bool take,int dummy=0>
    struct TakeHead {
        using N = typename TupleFilter<std::tuple<Tail...>,Filter>::T;
        using T = typename TuplePrepend<Head,N>::T;
    };
    template<int dummy>
    struct TakeHead<false,dummy> {
        using T = typename TupleFilter<std::tuple<Tail...>,Filter>::T;
    };

    // prop: on ne prend que si existe pas dans Tail
    using T = typename TakeHead<Filter::template Filter<Head>::keep>::T;
};
