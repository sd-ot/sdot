#ifndef SDOT_RECURSIVE_POLYTOP_HEADER
#define SDOT_RECURSIVE_POLYTOP_HEADER

#include "internal/RecursivePolytopImpl.h"
#include "../support/BumpPointerPool.h"
#include <deque>

/**

*/
template<class TF_,int dim_,class TI_=std::size_t,class UserData_=TI_>
class RecursivePolytop {
public:
    using                   UserData            = UserData_;
    enum {                  dim                 = dim_ };
    using                   TF                  = TF_;
    using                   TI                  = TI_;
    using                   Pt                  = Point<TF,dim>;

    struct                  Node                { Pt pos; UserData user_data; };
    using                   Impl                = RecursivePolytopImpl<RecursivePolytop,dim>;

    /**/                    RecursivePolytop    ();
    static RecursivePolytop convex_hull         ( const std::vector<Node> &nodes );

    void                    write_to_stream     ( std::ostream &os, std::string nl = "\n  ", std::string ns = "  " ) const;
    template<class VO> void display_vtk         ( VO &vo ) const;
    void                    plane_cut           ( Pt orig, Pt normal, const std::function<UserData(const UserData &,const UserData &,TF,TF)> &nf = {} );
    TF                      measure             () const;

    //template<class Nd> bool valid_node_prop   ( const std::vector<Nd> &prop, std::vector<Pt> prev_centers = {}, bool prev_centers_are_valid = true ) const;

public:
    BumpPointerPool         pool;
    Impl                    impl;
    mutable TI              date;
};

///**
//  Generic polytop (convex or not) defined by recursion
//*/
//template<class TF,int nvi_,int dim=nvi_,class TI=std::size_t,class NodeData=TI>
//class RecursivePolytop {
//public:
//    using                   Face               = RecursivePolytop<TF,nvi_-1,dim,TI,NodeData>;
//    using                   Node               = typename Face::Node;
//    enum {                  nvi                = nvi_ };
//    using                   Pt                 = Point<TF,dim>;
//    using                   VN                 = std::vector<Node>;
//    using                   DN                 = std::deque<Node>;

//    static RecursivePolytop convex_hull        ( const std::vector<Node> &nodes, const std::vector<Pt> &prev_centers = {} );

//    static std::vector<DN>  non_closed_node_seq( const std::vector<Face> &faces ); ///< get non closed sequence of nodes from faces. Works only for nvi == 2.
//    template<class Fu> void for_each_faces_rec ( const Fu &func ) const;
//    void                    write_to_stream    ( std::ostream &os, std::string sp = "\n  " ) const;
//    template<class Nd> bool valid_node_prop    ( const std::vector<Nd> &prop, std::vector<Pt> prev_centers = {}, bool prev_centers_are_valid = true ) const;
//    template<class Vk> void display_vtk        ( Vk &vtk_output ) const;
//    template<class Nd> auto with_nodes         ( const std::vector<Nd> &new_nodes ) const;
//    static RecursivePolytop make_from          ( const std::vector<Face> &faces );
//    RecursivePolytop        plane_cut          ( Pt orig, Pt normal, const std::function<Node(const Node &,const Node &,TF,TF)> &nf = {}, std::vector<Face> *new_faces = nullptr ) const;
//    bool                    operator<          ( const RecursivePolytop &that ) const;
//    std::vector<VN>         node_seq           () const; ///< make a sequence of nodes from faces. Works only for nvi == 2.
//    bool                    contains           ( const Pt &pt ) const;
//    TF                      measure            ( const std::vector<Pt> &prev_dirs = {}, TF div = 1 ) const;
//    TI                      max_id             () const;
//    operator                bool               () const { return nodes.size(); }

//    std::vector<Node>       nodes;
//    std::vector<Face>       faces;             ///<
//    std::vector<Node>       dirs;              ///< local base
//    std::string             name;
//};

///**
//  Segment
//*/
//template<class TF,int dim,class TI,class NodeData>
//class RecursivePolytop<TF,1,dim,TI,NodeData> {
//public:
//    using                   Pt                = Point<TF,dim>;
//    struct                  Node              { Pt pos; TI id; NodeData data; bool operator<( const Node &that ) const { return pos < that.pos; } bool operator==( const Node &that ) const { return pos == that.pos; } void write_to_stream( std::ostream &os ) const { os << "[" << pos << "]"; } };
//    using                   Face              = Node;
//    enum {                  nvi               = 1 };

//    static RecursivePolytop convex_hull       ( const std::vector<Node> &nodes, const std::vector<Pt> &prev_centers = {} );

//    template<class Fu> void for_each_faces_rec( const Fu &func ) const;
//    void                    write_to_stream   ( std::ostream &os, std::string sp = {} ) const;
//    template<class Nd> bool valid_node_prop   ( const std::vector<Nd> &prop, const std::vector<Pt> &prev_centers = {}, bool prev_centers_are_valid = true ) const;
//    template<class Nd> auto with_nodes        ( const std::vector<Nd> &new_nodes ) const;
//    static RecursivePolytop make_from         ( const std::vector<Face> &faces );
//    RecursivePolytop        plane_cut         ( Pt orig, Pt normal, const std::function<Node(const Node &,const Node &,TF,TF)> &nf, std::vector<Face> *new_faces = nullptr ) const;
//    bool                    operator<         ( const RecursivePolytop &that ) const;
//    bool                    contains          ( const Pt &pt ) const;
//    TF                      measure           ( const std::vector<Pt> &prev_dirs = {}, TF div = 1 ) const;
//    operator                bool              () const { return nodes.size(); }

//    std::vector<Node>       nodes;
//    std::vector<Node>       dirs;             ///< local base
//};

//template<class TF,int dim,class TI,class NodeData>
//bool operator<( const typename RecursivePolytop<TF,1,dim,TI,NodeData>::Node &a, const typename RecursivePolytop<TF,1,dim,TI,NodeData>::Node &b );

#include "RecursivePolytop.tcc"

#endif // SDOT_RECURSIVE_POLYTOP_HEADER
