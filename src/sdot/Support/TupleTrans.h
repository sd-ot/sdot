#pragma once

#include <tuple>

/**
  Allow to construct a tuple from a transformation or its sub-types
  Example of Transformer:
      struct MyTransformerToPtr {
          template<class U>
          struct Trans {
              using T = U *;
          };
      };

*/
template<class Tuple,class Transformer>
struct TupleTrans;

template<class... Items,class Transformer>
struct TupleTrans<std::tuple<Items...>,Transformer> {
    using T = std::tuple<typename Transformer::template Trans<Items>::T...>;
};
