#ifndef SDOT_RECURSIVE_POLYTOP_HEADER
#define SDOT_RECURSIVE_POLYTOP_HEADER

#include "Point.h"
#include <vector>

/**
  Generic polytop (convex or not) defined by recursion
*/
template<class TF,int nvi,int dim=nvi,class TI=std::size_t,class NodeData=TI>
class RecursivePolytop {
public:
    using                   Face           = RecursivePolytop<TF,nvi-1,dim,TI,NodeData>;
    using                   Node           = typename Face::Node;
    using                   Pt             = Point<TF,dim>;

    static RecursivePolytop convex_hull    ( const std::vector<Node> &nodes, const std::vector<Pt> &prev_centers = {} );

    void                    write_to_stream( std::ostream &os, std::string sp = "\n  " ) const;
    bool                    operator<      ( const RecursivePolytop &that ) const;

    std::vector<Node>       nodes;
    std::vector<Face>       faces;         ///<
    std::string             name;
};

/**
  Definition for a segment
*/
template<class TF,int dim,class TI,class NodeData>
class RecursivePolytop<TF,1,dim,TI,NodeData> {
public:
    using                   Pt             = Point<TF,dim>;
    struct                  Node           { Pt pos; NodeData data; bool operator<( const Node &that ) const { return pos < that.pos; } bool operator==( const Node &that ) const { return pos == that.pos; } };

    static RecursivePolytop convex_hull    ( const std::vector<Node> &nodes, const std::vector<Pt> &prev_centers = {} );

    void                    write_to_stream( std::ostream &os, std::string sp = {} ) const;
    bool                    operator<      ( const RecursivePolytop &that ) const;

    std::vector<Node>       nodes;
};

template<class TF,int dim,class TI,class NodeData>
bool operator<( const typename RecursivePolytop<TF,1,dim,TI,NodeData>::Node &a, const typename RecursivePolytop<TF,1,dim,TI,NodeData>::Node &b );

#include "RecursivePolytop.tcc"

#endif // SDOT_RECURSIVE_POLYTOP_HEADER
