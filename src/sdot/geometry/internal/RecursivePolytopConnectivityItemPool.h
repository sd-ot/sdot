#ifndef SDOT_RecursivePolytopConnectivityItemPool_HEADER
#define SDOT_RecursivePolytopConnectivityItemPool_HEADER

#include "RecursivePolytopConnectivityItem.h"
#include "../../support/BumpPointerPool.h"

/**
*/
template<class TF,class TI,int nvi>
struct RecursivePolytopConnectivityItemPool {
    using                        Next           = RecursivePolytopConnectivityItemPool<TF,TI,nvi-1>;
    using                        Item           = RecursivePolytopConnectivityItem<TF,TI,nvi>;
    using                        Face           = typename Item::Face;

    template<int n> auto         operator[]     ( N<n> v ) const { return next[ v ]; }
    template<int n> auto         operator[]     ( N<n> v ) { return next[ v ]; }
    auto                         operator[]     ( N<nvi> ) const { return this; }
    auto                         operator[]     ( N<nvi> ) { return this; }

    void                         write_to_stream( std::ostream &os ) const;
    Item*                        find_or_create ( BumpPointerPool &mem_pool, std::vector<Face *> &&sorted_faces ); ///< faces can sorted by adresses
    Item*                        create         ( BumpPointerPool &mem_pool, std::vector<Face *> &&sorted_faces );

    Item*                        last_in_pool   = nullptr;
    TI                           nb_items       = 0;
    Next                         next;
};

//
template<class TF,class TI>
struct RecursivePolytopConnectivityItemPool<TF,TI,0> {
    using                        Item           = RecursivePolytopConnectivityItem<TF,TI,0>;

    auto                         operator[]     ( N<0> ) const { return this; }
    auto                         operator[]     ( N<0> ) { return this; }

    void                         write_to_stream( std::ostream &os ) const;
    Item*                        find_or_create ( BumpPointerPool &mem_pool, TI node_number );
    Item*                        create         ( BumpPointerPool &mem_pool, TI node_number );

    Item*                        last_in_pool   = nullptr;
    TI                           nb_items       = 0;
};

#include "RecursivePolytopConnectivityItemPool.tcc"

#endif // SDOT_RecursivePolytopConnectivityItemPool_HEADER
