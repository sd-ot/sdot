#include "RecursivePolytopConnectivityItemPool.h"
#include <algorithm>

// find_or_create -----------------------------------------------------------------------------------------------------------------------------------------------------
template<class TF,class TI,int nvi>
RecursivePolytopConnectivityItem<TF,TI,nvi> *RecursivePolytopConnectivityItemPool<TF,TI,nvi>::find_or_create( BumpPointerPool &mem_pool, std::vector<Face *> &&sorted_faces ) {
    // already in the pool ?
    for( Item *item = last_in_pool; item; item = item->prev_in_pool )
        if ( item->faces == sorted_faces )
            return item;

    // else, create a new one
    return create( mem_pool, sorted_faces );
}

template<class TF,class TI>
RecursivePolytopConnectivityItem<TF,TI,0> *RecursivePolytopConnectivityItemPool<TF,TI,0>::find_or_create( BumpPointerPool &mem_pool, TI node_number ) {
    // already in the pool ?
    for( Item *item = last_in_pool; item; item = item->prev_in_pool )
        if ( item->node_number == node_number )
            return item;

    // else, create a new one
    return create( mem_pool, node_number );
}

// create ------------------------------------------------------------------------------------------------------------------------------------------------------------
template<class TF,class TI,int nvi>
RecursivePolytopConnectivityItem<TF,TI,nvi> *RecursivePolytopConnectivityItemPool<TF,TI,nvi>::create( BumpPointerPool &mem_pool, std::vector<Face *> &&sorted_faces ) {
    // make a sibling
    Item *alt = mem_pool.create<Item>();
    alt->prev_in_pool = last_in_pool;
    alt->num = nb_items++;
    last_in_pool = alt;

    alt->faces.resize( sorted_faces.size() );
    for( TI i = 0; i < alt->faces.size(); ++i )
        alt->faces[ i ] = sorted_faces[ i ]->sibling;
    std::sort( alt->faces.begin(), alt->faces.end() );

    // make the target item
    Item *res = mem_pool.create<Item>();
    res->prev_in_pool = last_in_pool;
    res->num = nb_items++;
    last_in_pool = res;

    res->faces = std::move( sorted_faces );

    // make the links
    res->sibling = alt;
    alt->sibling = res;

    return res;
}

template<class TF,class TI>
RecursivePolytopConnectivityItem<TF,TI,0> *RecursivePolytopConnectivityItemPool<TF,TI,0>::create( BumpPointerPool &mem_pool, TI node_number ) {
    // make a sibling
    Item *alt = mem_pool.create<Item>();
    alt->prev_in_pool = last_in_pool;
    alt->node_number = node_number;
    alt->num = nb_items++;
    last_in_pool = alt;

    // make the target item
    Item *res = mem_pool.create<Item>();
    res->prev_in_pool = last_in_pool;
    res->node_number = node_number;
    res->num = nb_items++;
    last_in_pool = res;

    // make the links
    res->sibling = alt;
    alt->sibling = res;

    return res;
}

// create ------------------------------------------------------------------------------------------------------------------------------------------------------------
template<class TF,class TI,int nvi>
void RecursivePolytopConnectivityItemPool<TF,TI,nvi>::write_to_stream( std::ostream &os ) const {
    next.write_to_stream( os );

    os << "nvi " << nvi << ":";
    for( Item *item = last_in_pool; item; item = item->prev_in_pool ) {
        if ( item->num % 2 )
            continue;
        os << "\n  " << item->num / 2 << ":";
        for( auto *face : item->sorted_faces )
            os << " " << face->num / 2;
    }
}

template<class TF,class TI>
void RecursivePolytopConnectivityItemPool<TF,TI,0>::write_to_stream( std::ostream &/*os*/ ) const {
}

