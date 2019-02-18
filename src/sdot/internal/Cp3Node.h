#pragma once

#include "../system/ListRef.h"
#include "../Point3.h"
#include <ostream>

namespace sdot {

template<class Pc>
struct Cp3Edge;

/**
*/
template<class Pc>
struct Cp3Node {
    struct   GetNextEdge    { Cp3Edge<Pc> *&operator()( Cp3Edge<Pc> *e ) const { return e->next_from_same_node; } };
    using    EdgeList       = ListRef<Cp3Edge<Pc>,GetNextEdge>;     ///<
    using    Edge           = Cp3Edge<Pc>;     ///<
    using    TF             = typename Pc::TF; ///< floating point type
    using    TI             = typename Pc::TI; ///< size type
    using    Pt             = Point3<TF>;      ///< 3D point
    union    Ni             { Cp3Node *node; Edge *edge; };

    void     write_to_stream( std::ostream &os ) const { os << pos; }

    Cp3Node *prev_in_pool;  ///<
    Cp3Node *next_in_pool;  ///<
    Cp3Node *next_in_cut;   ///<
    TI       resp_bound;    ///<
    TI       op_count;      ///<
    EdgeList edges;         ///<
    Ni       nitem;         ///< new node or edge for current operation
    Pt       pos;           ///<
};

} // namespace sdot
