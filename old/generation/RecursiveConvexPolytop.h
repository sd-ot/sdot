#ifndef SDOT_RECURSIVE_POLYTOP_HEADER
#define SDOT_RECURSIVE_POLYTOP_HEADER

#include "internal/RecursivePolytopConnectivity.h"


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

    /**/                            RecursiveConvexPolytop( const std::vector<Pt> &vertices );

    RecursiveConvexPolytop          plane_cut             ( Pt orig, Pt normal ) const;

    template<class F,int n> void    for_each_item_rec    ( const F &fu, N<n> ) const; ///< for a fixed nvi
    template<class F> void          for_each_item_rec    ( const F &fu ) const;

    void                            write_to_stream      ( std::ostream &os, std::string nl = "\n  ", std::string ns = "  " ) const;
    template<class VO> void         display_vtk          ( VO &vo ) const;

    bool                            all_vertices_are_used() const;
    TI                              nb_vertices          () const { return vertices.size(); }
    bool                            contains             ( const Pt &pt ) const;
    TF                              measure              () const;
    const Pt&                       vertex               ( TI i ) const { return vertices[ i ]; }

private:
    template<class a,int b,int c,class d> friend struct  RecursivePolytopConnectivity;
    using                           Impl                 = RecursivePolytopConnectivity<TF,dim,dim,TI>;

    std::vector<Pt>                 vertices;            ///< points
    std::vector<Impl>               impls;               ///< volumes
};

#include "RecursivePolytop.tcc"

#endif // SDOT_RECURSIVE_POLYTOP_HEADER
