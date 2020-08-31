#include "RecursivePolytopConnectivityPool.h"

template<class TF,class TI,int nvi> template<int n,class F>
void RecursivePolytopConnectivityPool<TF,TI,nvi>::get_item( RecursivePolytopConnectivityItem<TF,TI,n> *&res, bool &neg, BumpPointerPool &pool, const F &sorted_faces ) {
    next.get_item( res, neg, pool, sorted_faces );
}

template<class TF,class TI,int nvi>
void RecursivePolytopConnectivityPool<TF,TI,nvi>::get_item( Item *&res, bool &neg, BumpPointerPool &pool, const std::vector<typename Item::OrientedFace> &sorted_faces ) {
    // already in the pool ?
    for( Item *item = last_in_pool; item; item = item->prev_in_pool ) {
        if ( item->faces.size() != sorted_faces.size() )
            continue;
        for( TI i = 0; ; ++i ) {
            if ( i == sorted_faces.size() ) {
                neg = item->faces[ 0 ].neg != sorted_faces[ 0 ].neg;
                res = item;
                return;
            }
            if ( item->faces[ i ].ref != sorted_faces[ i ].ref )
                break;
        }
    }

    // else, create a new one
    res = pool.create<Item>();
    res->prev_in_pool = last_in_pool;
    res->faces = sorted_faces;
    last_in_pool = res;

    neg = false;
}


template<class TF,class TI>
void RecursivePolytopConnectivityPool<TF,TI,0>::get_item( N<0>, Item *&res, bool &neg, BumpPointerPool &pool, TI node_number ) {
    res = pool.create<Item>();
    res->prev_in_pool = last_in_pool;
    res->node_number = node_number;
    last_in_pool = res;

    neg = false;
}
