#ifndef SDOT_RecursivePolytopConnectivityPool_HEADER
#define SDOT_RecursivePolytopConnectivityPool_HEADER

#include "RecursivePolytopConnectivityItem.tcc"
#include "../../support/BumpPointerPool.h"

/**
*/
template<class TF,class TI,int nvi>
struct RecursivePolytopConnectivityPool {
    using                        Next         = RecursivePolytopConnectivityPool<TF,TI,nvi-1>;
    using                        Item         = RecursivePolytopConnectivityItem<TF,TI,nvi>;

    template<int n,class F> void get_item     ( RecursivePolytopConnectivityItem<TF,TI,n> *&res, bool &neg, BumpPointerPool &pool, const F &sorted_faces ); ///< faces can sorted by adresses
    void                         get_item     ( Item *&res, bool &neg, BumpPointerPool &pool, const std::vector<typename Item::OrientedFace> &sorted_faces ); ///< faces can sorted by adresses

    Item*                        last_in_pool = nullptr;
    Next                         next;
};

//
template<class TF,class TI>
struct RecursivePolytopConnectivityPool<TF,TI,0> {
    using                        Item         = RecursivePolytopConnectivityItem<TF,TI,0>;

    void                         get_item     ( N<0>, Item *&item, bool &neg, BumpPointerPool &pool, TI node_number );

    Item*                        last_in_pool = nullptr;
};

#include "RecursivePolytopConnectivityPool.tcc"

#endif // SDOT_RecursivePolytopConnectivityPool_HEADER
