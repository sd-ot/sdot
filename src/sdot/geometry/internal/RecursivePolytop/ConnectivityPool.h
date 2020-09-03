#ifndef SDOT_RecursivePolytopConnectivityItemPool_HEADER
#define SDOT_RecursivePolytopConnectivityItemPool_HEADER

#include "Connectivity.h"

namespace sdot {
namespace internal {
namespace RecursivePolytop {

/**
*/
template<class TI,int nvi>
struct ConnectivityPool {
    using                  Cnn            = Connectivity<TI,nvi>;
    using                  Bnd            = Connectivity<TI,nvi-1>;
    using                  Ocn            = OrientedConnectivity<Cnn>;
    using                  Obn            = OrientedConnectivity<Bnd>;
    using                  Nxt            = ConnectivityPool<TI,nvi-1>;

    template<int n> auto   operator[]     ( N<n> v ) const { return next[ v ]; }
    template<int n> auto   operator[]     ( N<n> v ) { return next[ v ]; }
    auto                   operator[]     ( N<nvi> ) const { return this; }
    auto                   operator[]     ( N<nvi> ) { return this; }

    void                   write_to_stream( std::ostream &os ) const;
    template<class F> void apply_rec      ( const F &f ) const;

    Ocn                    find_or_create ( BumpPointerPool &mem_pool, std::vector<Obn> &&sorted_boundaries );
    Cnn*                   create         ( BumpPointerPool &mem_pool, std::vector<Obn> &&sorted_boundaries );

    Cnn*                   last_in_pool   = nullptr;
    Nxt                    next;
};

//
template<class TI>
struct ConnectivityPool<TI,0> {
    using                  Cnn            = Connectivity<TI,0>;
    using                  Ocn            = OrientedConnectivity<Cnn>;

    auto                   operator[]     ( N<0> ) const { return this; }
    auto                   operator[]     ( N<0> ) { return this; }

    void                   write_to_stream( std::ostream &os ) const;
    template<class F> void apply_rec      ( const F &f ) const;

    Ocn                    find_or_create ( BumpPointerPool &mem_pool, TI node_number, bool neg );
    Cnn*                   create         ( BumpPointerPool &mem_pool, TI node_number );

    Cnn*                   last_in_pool   = nullptr;
};

} // namespace sdot
} // namespace internal
} // namespace RecursivePolytop

#include "ConnectivityPool.tcc"

#endif // SDOT_RecursivePolytopConnectivityItemPool_HEADER
