#ifndef SDOT_generation_Shape_H
#define SDOT_generation_Shape_H

#include "../geometry/Point.h"
#include "Rational.h"

namespace sdot {

/**
*/
template<int dim,int nvi>
struct Shape {
    using TF             = Rational;
    using Pt             = Point<TF,dim>;

    void  write_to_stream( std::ostream &os ) const {}
};

/**
*/
template<int dim>
struct Shape<dim,2> {
    using TF             = Rational;
    using Pt             = Point<TF,dim>;

    void  write_to_stream( std::ostream &os ) const {}


};

} // namespace sdot

#include "Shape.tcc"

#endif // SDOT_generation_Shape_H
