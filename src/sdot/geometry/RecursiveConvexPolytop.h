#ifndef SDOT_RECURSIVE_POLYTOP_HEADER
#define SDOT_RECURSIVE_POLYTOP_HEADER

#include "internal/RecursivePolytop/Pool.h"


/**
  Starting from a set of point, make_convex_hull adds faces
*/
template<class TF_,int dim_,class TI_=std::size_t>
class RecursiveConvexPolytop {
public:
    using                           ItemPool              = RecursivePolytopConnectivityItemPool<TF_,TI_,dim_>;
    using                           Vertex                = RecursivePolytopConnectivityItem<TF_,TI_,0>;
    using                           Item                  = RecursivePolytopConnectivityItem<TF_,TI_,dim_>;
    enum {                          dim                   = dim_ };
    using                           TF                    = TF_;
    using                           TI                    = TI_;
    using                           Pt                    = Point<TF,dim>;
    using                           Rp                    = RecursiveConvexPolytop;

    /**/                            RecursiveConvexPolytop( const std::vector<Pt> &old_positions, const ItemPool &old_item_pool, const std::vector<Item *> &old_items ); ///< make a copy (with only the necessary items)
    /**/                            RecursiveConvexPolytop( std::vector<Pt> &&positions = {} ); ///< make a convex hull from the nodes (if non empty)
    /**/                            RecursiveConvexPolytop( const RecursiveConvexPolytop &that );
    /**/                            RecursiveConvexPolytop( RecursiveConvexPolytop &&that );

    void                            write_to_stream       ( std::ostream &os ) const;
    template<class VO> void         display_vtk           ( VO &vo ) const;
    Rp&                             operator=             ( RecursiveConvexPolytop &&that );
    Rp                              plane_cut             ( Pt orig, Pt normal ) const; ///< a cut based on convexity
    std::vector<Rp>                 conn_cut              ( Pt orig, Pt normal ) const; ///< a plane cut based on connectivity. Return a list of possibilities

private:
    void                            _make_convex_hull     ();

    std::vector<Pt>                 positions;            ///<
    ItemPool                        item_pool;            ///<
    BumpPointerPool                 mem_pool;             ///<
};

#include "RecursiveConvexPolytop.tcc"

#endif // SDOT_RECURSIVE_POLYTOP_HEADER
