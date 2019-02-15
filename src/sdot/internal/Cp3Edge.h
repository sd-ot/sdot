#pragma once

#include "Cp3Node.h"

namespace sdot {

template<class Pc>
struct Cp3Face;

/**
*/
template<class Pc>
struct Cp3Edge {
    using    Node                 = Cp3Node<Pc>;     ///<
    using    Face                 = Cp3Face<Pc>;     ///<
    using    TF                   = typename Pc::TF; ///< floating point type
    using    TI                   = typename Pc::TI; ///< size type
    using    Pt                   = Point3<TF>;      ///< 3D point

    /**/     Cp3Edge              () : op_count( 0 ) {}

    void     write_to_stream      ( std::ostream &os ) const { os << "Edge(" << *n0 << "," << *n1 << ")"; }

    bool     straight             () const { return radius < 0; }
    bool     round                () const { return radius >= 0; }
    Pt       X                    () const { return ndir; }
    Pt       Y                    () const { return tangent_0; }

    // input data
    Cp3Edge *next_from_same_node; ///<
    Cp3Edge *next_from_same_face; ///<
    Cp3Edge *prev_in_pool;        ///<
    Cp3Edge *next_in_cut;         ///<
    TI       op_count;            ///<
    Cp3Edge *sibling;             ///< same edge (maybe with a different orientation), but attached to an other face
    Cp3Edge *nedge;               ///< new edge for current operation
    Face    *face;                ///< attached face
    Node    *n0;                  ///< index of 1st node
    Node    *n1;                  ///< index of 2nd node

    // computed data (for round edges)
    Pt       tangent_0;           ///< tangent in n0
    Pt       tangent_1;           ///< tangent in n1
    TF       angle_1;             ///< angle of n1 (angle of n0 = 0)
    Pt       center;              ///<
    TF       radius;              ///<
    Pt       ndir;                ///< normalized( node[ n0 ].pos - center )
};

} // namespace sdot
