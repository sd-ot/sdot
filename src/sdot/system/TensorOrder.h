#pragma once

// #include "Vec.h"
#include <vector>
#include <array>

/**
  @brief To get the static dimensionnality of objects

  Meaning of \c res
   - 0 -> scalar
   - 1 -> vector
   - 2 -> matrix
   - ...
   - -1 -> dynamic tensor order (not fixed during the compilation)
*/
template<class T> struct TensorOrder { static const int res = 0; };

template<class T,class A>       struct TensorOrder<std::vector<T,A>> { static const int res = 1; };
template<class T,std::size_t n> struct TensorOrder<std::array<T,n>>  { static const int res = 1; };
//template<class T> struct TensorOrder<Vec        <T>> { static const int res = 1; };
