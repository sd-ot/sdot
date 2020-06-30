#ifndef SDOT_RECURSIVE_POLYTOP_HEADER
#define SDOT_RECURSIVE_POLYTOP_HEADER

#include "Point.h"
#include <vector>

/**
  Generic polytop (convex or not) defined by recursion
*/
template<class TF,int nvi_,int dim=nvi_,class TI=std::size_t,class NodeData=TI>
class RecursivePolytop {
public:
    using                   Face              = RecursivePolytop<TF,nvi_-1,dim,TI,NodeData>;
    using                   Node              = typename Face::Node;
    enum {                  nvi               = nvi_ };
    using                   Pt                = Point<TF,dim>;

    static RecursivePolytop convex_hull       ( const std::vector<Node> &nodes, const std::vector<Pt> &prev_centers = {} );

    template<class Fu> void for_each_faces_rec( const Fu &func ) const;
    void                    write_to_stream   ( std::ostream &os, std::string sp = "\n  " ) const;
    template<class Nd> bool valid_node_prop   ( const std::vector<Nd> &prop, std::vector<Pt> prev_centers = {}, bool prev_centers_are_valid = true ) const;
    template<class Vk> void display_vtk       ( Vk &vtk_output ) const;
    template<class Nd> auto with_nodes        ( const std::vector<Nd> &new_nodes ) const;
    void                    plane_cut         ( std::vector<RecursivePolytop> &res, Pt orig, Pt normal ) const;
    TF                      measure           ( const std::vector<Pt> &prev_dirs = {}, TF div = 1 ) const;
    bool                    operator<         ( const RecursivePolytop &that ) const;

    std::vector<Node>       nodes;
    std::vector<Face>       faces;            ///<
    std::string             name;
};

/**
  Definition for a segment
*/
template<class TF,int dim,class TI,class NodeData>
class RecursivePolytop<TF,1,dim,TI,NodeData> {
public:
    using                   Pt                = Point<TF,dim>;
    struct                  Node              { Pt pos; NodeData data; bool operator<( const Node &that ) const { return pos < that.pos; } bool operator==( const Node &that ) const { return pos == that.pos; } void write_to_stream( std::ostream &os ) const { os << data; } };
    enum {                  nvi               = 1 };

    static RecursivePolytop convex_hull       ( const std::vector<Node> &nodes, const std::vector<Pt> &prev_centers = {} );

    template<class Fu> void for_each_faces_rec( const Fu &func ) const;
    void                    write_to_stream   ( std::ostream &os, std::string sp = {} ) const;
    template<class Nd> bool valid_node_prop   ( const std::vector<Nd> &prop, const std::vector<Pt> &prev_centers = {}, bool prev_centers_are_valid = true ) const;
    template<class Nd> auto with_nodes        ( const std::vector<Nd> &new_nodes ) const;
    void                    plane_cut         ( std::vector<RecursivePolytop> &res, Pt orig, Pt normal ) const;
    TF                      measure           ( const std::vector<Pt> &prev_dirs = {}, TF div = 1 ) const;
    bool                    operator<         ( const RecursivePolytop &that ) const;

    std::vector<Node>       nodes;
};

template<class TF,int dim,class TI,class NodeData>
bool operator<( const typename RecursivePolytop<TF,1,dim,TI,NodeData>::Node &a, const typename RecursivePolytop<TF,1,dim,TI,NodeData>::Node &b );

#include "RecursivePolytop.tcc"

#endif // SDOT_RECURSIVE_POLYTOP_HEADER
