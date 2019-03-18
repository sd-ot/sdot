#pragma once

#include "Cp3Edge.h"

namespace sdot {

/**
*/
template<class Pc>
struct Cp3Face {
    struct   GetNextEdge    { Cp3Edge<Pc> *&operator()( Cp3Edge<Pc> *e ) const { return e->next_from_same_face; } };
    using    EdgeList       = ListRef<Cp3Edge<Pc>,GetNextEdge>;     ///<
    using    Edge           = Cp3Edge<Pc>;     ///<
    using    TF             = typename Pc::TF; ///< floating point type
    using    TI             = typename Pc::TI; ///< size type
    using    CI             = typename Pc::CI; ///< cut info
    using    Pt             = Point3<TF>;      ///< 3D point

    void     write_to_stream( std::ostream &os ) const { edges.write_to_stream( os << "Face(", ", " ); os << ")"; }

    Cp3Face *prev_in_pool;  ///<
    Cp3Face *next_in_pool;  ///<
    Cp3Face *next_in_cut;   ///<
    TI       op_count;      ///<
    CI       cut_id;        ///< provided by the user (as argument of the cut function)
    Pt       cut_O;         ///< a point in the cut plane
    Pt       cut_N;         ///< normal, oriented toward exterior, i.e. the removed part (if the face is actually plane)
    EdgeList edges;         ///<
    bool     round;         ///<
};

} // namespace sdot

