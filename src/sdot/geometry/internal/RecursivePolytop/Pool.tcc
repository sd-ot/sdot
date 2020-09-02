#include <algorithm>
#include "Pool.h"

namespace sdot {
namespace internal {
namespace RecursivePolytop {

// find_or_create -----------------------------------------------------------------------------------------------------------------------------------------------------
template<class TF,class TI,int nvi>
RecursivePolytopConnectivityItem<TF,TI,nvi> *Pool<TF,TI,nvi>::find_or_create( BumpPointerPool &mem_pool, std::vector<Face *> &&sorted_faces ) {
    // already in the pool ?
    for( Item *item = last_in_pool; item; item = item->prev_in_pool )
        if ( item->faces == sorted_faces )
            return item;

    // else, create a new one
    return create( mem_pool, std::move( sorted_faces ) );
}

template<class TF,class TI>
RecursivePolytopConnectivityItem<TF,TI,0> *Pool<TF,TI,0>::find_or_create( BumpPointerPool &mem_pool, TI node_number ) {
    // already in the pool ?
    for( Item *item = last_in_pool; item; item = item->prev_in_pool )
        if ( item->node_number == node_number )
            return item;

    // else, create a new one
    return create( mem_pool, node_number );
}

// create ------------------------------------------------------------------------------------------------------------------------------------------------------------
template<class TF,class TI,int nvi>
RecursivePolytopConnectivityItem<TF,TI,nvi> *Pool<TF,TI,nvi>::create( BumpPointerPool &mem_pool, std::vector<Face *> &&sorted_faces ) {
    Item *res = mem_pool.create<Item>();
    res->faces = std::move( sorted_faces );
    res->prev_in_pool = last_in_pool;
    last_in_pool = res;

    return res;
}

template<class TF,class TI>
RecursivePolytopConnectivityItem<TF,TI,0> *Pool<TF,TI,0>::create( BumpPointerPool &mem_pool, TI node_number ) {
    Item *res = mem_pool.create<Item>();
    res->prev_in_pool = last_in_pool;
    res->node_number = node_number;
    last_in_pool = res;

    return res;
}

// create ------------------------------------------------------------------------------------------------------------------------------------------------------------
template<class TF,class TI,int nvi>
void Pool<TF,TI,nvi>::write_to_stream( std::ostream &os ) const {
    next.write_to_stream( os );

    os << "\n  nvi " << nvi << ":";
    for( const Item *item = last_in_pool; item; item = item->prev_in_pool ) {
        os << "\n    " << item->num << ":";
        for( const auto &face : item->faces )
            face.write_to_stream( os << " " );
    }
}

template<class TF,class TI>
void Pool<TF,TI,0>::write_to_stream( std::ostream &os ) const {
    os << "\n  nvi 0:";
    for( Item *item = last_in_pool; item; item = item->prev_in_pool )
        os << "\n    " << item->num << ": " << item->node_number;
}


// apply_rec -----------------------------------------------------------------------------------------------------------------------------------------------------
template<class TF,class TI,int nvi> template<class F>
void Pool<TF,TI,nvi>::apply_rec( const F &f ) const {
    next.apply_rec( f );
    for( Item *item = last_in_pool; item; item = item->prev_in_pool )
        f( item );
}

template<class TF,class TI> template<class F>
void Pool<TF,TI,0>::apply_rec( const F &f ) const {
    for( Item *item = last_in_pool; item; item = item->prev_in_pool )
        f( item );
}

} // namespace sdot
} // namespace internal
} // namespace RecursivePolytop
