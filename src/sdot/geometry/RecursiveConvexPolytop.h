#ifndef SDOT_RECURSIVE_POLYTOP_HEADER
#define SDOT_RECURSIVE_POLYTOP_HEADER

#include "internal/RecursivePolytop/ConnectivityPool.h"
#include "Point.h"

namespace sdot {

/**
  Starting from a set of point, make_convex_hull adds faces
*/
template<class TF_,int dim_,class TI_=std::size_t>
class RecursiveConvexPolytop {
public:
    using                           ConnectivityPool      = internal::RecursivePolytop::ConnectivityPool<TI_,dim_>;
    using                           OrientedItem          = internal::RecursivePolytop::OrientedConnectivity<internal::RecursivePolytop::Connectivity<TI_,dim_>>;
    using                           Vertex                = internal::RecursivePolytop::Connectivity<TI_,0>;
    using                           Item                  = internal::RecursivePolytop::Connectivity<TI_,dim_>;
    enum {                          dim                   = dim_ };
    using                           TF                    = TF_;
    using                           TI                    = TI_;
    using                           Pt                    = Point<TF,dim>;
    using                           Rp                    = RecursiveConvexPolytop;

    /**/                            RecursiveConvexPolytop( const std::vector<Pt> &old_positions, const ConnectivityPool &old_item_pool, const std::vector<OrientedItem> &old_items ); ///< make a copy (with only the necessary items)
    /**/                            RecursiveConvexPolytop( const RecursiveConvexPolytop &that );
    /**/                            RecursiveConvexPolytop( RecursiveConvexPolytop &&that );
    /**/                            RecursiveConvexPolytop( std::vector<Pt> &&positions ); ///< make a convex hull from the nodes
    /**/                            RecursiveConvexPolytop();

    Rp&                             operator=             ( const RecursiveConvexPolytop &that );
    Rp&                             operator=             ( RecursiveConvexPolytop &&that );

    void                            write_to_stream       ( std::ostream &os ) const;
    template<class VO> void         display_vtk           ( VO &vo ) const;
    Rp                              plane_cut             ( Pt orig, Pt normal ) const; ///< a cut based on convexity
    std::vector<Rp>                 conn_cut              ( Pt orig, Pt normal ) const; ///< a plane cut based on connectivity. Return a list of possibilities

private:
    void                            _make_convex_hull     ();

    ConnectivityPool                connectivity_pool;    ///<
    std::vector<Pt>                 positions;            ///<
    BumpPointerPool                 mem_pool;             ///<
    std::vector<OrientedItem>       items;                ///<
};

} // namespace sdot

#include "RecursiveConvexPolytop.tcc"

#endif // SDOT_RECURSIVE_POLYTOP_HEADER
