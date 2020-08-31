#ifndef SDOT_RECURSIVE_POLYTOP_HEADER
#define SDOT_RECURSIVE_POLYTOP_HEADER

#include "internal/RecursivePolytopConnectivityPool.h"


/**
  Starting from a set of point, make_convex_hull adds faces
*/
template<class TF_,int dim_,class TI_=std::size_t>
class RecursiveConvexPolytop {
public:
    enum {                          dim                   = dim_ };
    using                           TF                    = TF_;
    using                           TI                    = TI_;
    using                           Pt                    = Point<TF,dim>;
    using                           Rp                    = RecursiveConvexPolytop;
    using                           Item                  = RecursivePolytopConnectivityItem<TF,TI,dim>;
    using                           ItemPool              = RecursivePolytopConnectivityPool<TF,TI,dim>;

    /**/                            RecursiveConvexPolytop( std::vector<Pt> &&positions = {} ); ///< make a convex hull from the nodes (if non empty)

    std::vector<Item *>             connected_connectivities;
    std::vector<Pt>                 positions;
    ItemPool                        item_pool;
    BumpPointerPool                 pool;

    //    using                           VVRp                  = std::vector<std::vector<Rp>>;
    //    RecursiveConvexPolytop          plane_cut             ( Pt orig, Pt normal ) const; ///< a cut based on convexity
    //    VVRp                            conn_cut              ( Pt orig, Pt normal ) const; ///< a cut based on connectivity and that does not rely on convexity. Return a list of possible results, each one consisting in a set of disjoint Rp

    //    template<class F,int n> void    for_each_item_rec    ( const F &fu, N<n> ) const; ///< for a fixed nvi
    //    template<class F> void          for_each_item_rec    ( const F &fu ) const;

    //    void                            write_to_stream      ( std::ostream &os, std::string nl = "\n  ", std::string ns = "  " ) const;
    //    template<class VO> void         display_vtk          ( VO &vo ) const;

    //    bool                            all_vertices_are_used() const;
    //    TI                              nb_vertices          () const { return nodes.size(); }
    //    bool                            contains             ( const Pt &pt ) const;
    //    TF                              measure              () const;
    //    const Pt&                       vertex               ( TI i ) const { return nodes[ i ]; }

    //private:
    //    template<class a,int b,int c,class d> friend struct  RecursivePolytopConnectivity;
    //    using                           Connectivity         = RecursivePolytopConnectivity<TF,dim,dim,TI>;

    //    Connectivity                    connectivity;        ///<
    //    std::vector<Pt>                 nodes;               ///<
};

#include "RecursiveConvexPolytop.tcc"

#endif // SDOT_RECURSIVE_POLYTOP_HEADER
