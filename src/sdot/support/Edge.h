#ifndef SDOT_EDGE_H
#define SDOT_EDGE_H

#include "Point.h"

namespace sdot {

/**
*/
template<class TF,int dim>
struct Edge {
    /**/          Edge           ( Point<TF,dim> a, Point<TF,dim> b ) : points{ a, b } {}
    /**/          Edge           () {}

    void          write_to_stream( std::ostream &os ) const { points[ 0 ].write_to_stream( os << "[" ); points[ 1 ].write_to_stream( os << "," ); os << "]"; }

    Point<TF,dim> points[ 2 ];
};

} // namespace sdot

#endif // SDOT_EDGE_H
