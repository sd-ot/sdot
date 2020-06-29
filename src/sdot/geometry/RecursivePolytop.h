#ifndef SDOT_RECURSIVE_POLYTOP_HEADER
#define SDOT_RECURSIVE_POLYTOP_HEADER

#include "Point.h"
#include <vector>

/**
  Slow generic polytop (convex or not) defined by recursion
*/
template<class TF,int dim,class TI=std::size_t>
class RecursivePolytop {
public:
    using  Pt   = Point<TF,dim>;
    struct Node { Pt pos; };


};

#include "RecursivePolytop.tcc"

#endif // SDOT_RECURSIVE_POLYTOP_HEADER
