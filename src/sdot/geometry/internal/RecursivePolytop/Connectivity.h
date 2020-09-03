#ifndef SDOT_RecursivePolytopConnectivityItem_HEADER
#define SDOT_RecursivePolytopConnectivityItem_HEADER

#include "../../../support/BumpPointerPool.h"
#include "../../../support/Void.h"
#include "../../../support/N.h"

#include "OrientedConnectivity.h"

#include <functional>
#include <vector>

namespace sdot {
namespace internal {
namespace RecursivePolytop {

template<class TI,int nvi>
struct ConnectivityPool;

/**
*/
template<class TI_,int nvi_>
struct Connectivity {
    using                          Obn            = OrientedConnectivity<Connectivity<TI_,nvi_-1>>;
    using                          Vvc            = std::vector<std::vector<Connectivity *>>;
    using                          Ocn            = OrientedConnectivity<Connectivity>;
    using                          Cpl            = ConnectivityPool<TI_,nvi_>;
    using                          Bnd            = Connectivity<TI_,nvi_-1>;
    using                          Vtx            = Connectivity<TI_,0>;
    using                          Mpl            = BumpPointerPool;
    using                          Cnn            = Connectivity;
    enum {                         nvi            = nvi_ };
    using                          TI             = TI_;

    template<class Pt> static void add_convex_hull( std::vector<Ocn> &res, Cpl &item_pool, Mpl &mem_pool, const Pt *positions, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &center );

    void                           write_to_stream( std::ostream &os ) const;
    const Vtx*                     first_vertex   () const { return boundaries[ 0 ].connectivity->first_vertex(); }
    Vtx*                           first_vertex   () { return boundaries[ 0 ].connectivity->first_vertex(); }
    template<class Pt> Cnn*        copy_rec       ( std::vector<Pt> &new_positions, Cpl &new_item_pool, Mpl &new_mem_pool, const std::vector<Pt> &old_positions ) const;
    template<int n> void           conn_cut       ( Cpl &new_item_pool, Mpl &new_mem_pool, N<n>, const std::function<TI(TI,TI)> &interp ) const;
    void                           conn_cut       ( Cpl &new_item_pool, Mpl &new_mem_pool, N<2>, const std::function<TI(TI,TI)> &interp ) const;
    void                           conn_cut       ( Cpl &new_item_pool, Mpl &new_mem_pool, N<1>, const std::function<TI(TI,TI)> &interp ) const;

    Cnn*                           prev_in_pool;  ///<
    std::vector<Obn>               boundaries;    ///<

    mutable Vvc                    new_items;     ///< list of possibilities, each one potentially containing several sub-items
    mutable Cnn*                   new_item;      ///< used for copies
    mutable TI                     tmp_num;       ///<
};

/** Vertex */
template<class TI_>
struct Connectivity<TI_,0> {
    using                          Vvc            = std::vector<std::vector<Connectivity *>>;
    using                          Ocn            = OrientedConnectivity<Connectivity>;
    using                          Cpl            = ConnectivityPool<TI_,0>;
    using                          Mpl            = BumpPointerPool;
    using                          Cnn            = Connectivity;
    enum {                         nvi            = 0 };
    using                          TI             = TI_;

    template<class Pt> static void add_convex_hull( std::vector<Ocn> &res, Cpl &item_pool, Mpl &mem_pool, const Pt *positions, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &center );

    void                           write_to_stream( std::ostream &os ) const;
    const Connectivity*            first_vertex   () const { return this; }
    Connectivity*                  first_vertex   () { return this; }
    template<class Pt> Cnn*        copy_rec       ( std::vector<Pt> &new_positions, Cpl &new_item_pool, Mpl &new_mem_pool, const std::vector<Pt> &old_positions ) const;
    void                           conn_cut       ( Cpl &new_item_pool, Mpl &new_mem_pool, N<0>, const std::function<TI(TI,TI)> &interp ) const;

    Cnn*                           prev_in_pool;  ///<
    TI                             node_number;   ///<

    mutable Vvc                    new_items;      ///< list of possibilities, each one potentially containing several sub-items
    mutable Cnn*                   new_item;      ///< used for copies
    mutable TI                     tmp_num;       ///<
};

} // namespace sdot
} // namespace internal
} // namespace RecursivePolytop

#include "Connectivity.tcc"

#endif // SDOT_RecursivePolytopConnectivityItem_HEADER
