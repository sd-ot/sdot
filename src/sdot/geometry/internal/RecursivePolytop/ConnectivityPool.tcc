#include <algorithm>
#include "ConnectivityPool.h"

namespace sdot {
namespace internal {
namespace RecursivePolytop {

// find_or_create -----------------------------------------------------------------------------------------------------------------------------------------------------
template<class TI,int nvi>
typename ConnectivityPool<TI,nvi>::Ocn ConnectivityPool<TI,nvi>::find_or_create( BumpPointerPool &mem_pool, std::vector<Obn> &&sorted_boundaries ) {
    // already in the pool ?
    for( Cnn *item = last_in_pool; item; item = item->prev_in_pool ) {
        if ( item->boundaries.size() != sorted_boundaries.size() )
            continue;

        bool neg_ok = true;
        bool pos_ok = true;
        for( TI i = 0; ; ++i ) {
            if ( i == sorted_boundaries.size() ) {
                if ( pos_ok )
                    return { item, false };
                if ( neg_ok )
                    return { item, true };
                break;
            }
            if ( item->boundaries[ i ].connectivity != sorted_boundaries[ i ].connectivity )
                break;
            neg_ok &= item->boundaries[ i ].neg != sorted_boundaries[ i ].neg;
            pos_ok &= item->boundaries[ i ].neg == sorted_boundaries[ i ].neg;
        }
    }

    // else, create a new one
    return { create( mem_pool, std::move( sorted_boundaries ) ), false };
}

template<class TI>
typename ConnectivityPool<TI,0>::Ocn ConnectivityPool<TI,0>::find_or_create( BumpPointerPool &mem_pool, TI node_number, bool neg ) {
    // already in the pool ?
    for( Cnn *item = last_in_pool; item; item = item->prev_in_pool )
        if ( item->node_number == node_number )
            return { item, neg };

    // else, create a new one
    return { create( mem_pool, node_number ), neg };
}

// create ------------------------------------------------------------------------------------------------------------------------------------------------------------
template<class TI,int nvi>
typename ConnectivityPool<TI,nvi>::Cnn *ConnectivityPool<TI,nvi>::create( BumpPointerPool &mem_pool, std::vector<Obn> &&sorted_boundaries ) {
    Cnn *res = mem_pool.create<Cnn>();
    res->prev_in_pool = last_in_pool;
    last_in_pool = res;

    res->boundaries = std::move( sorted_boundaries );

    return res;
}

template<class TI>
typename ConnectivityPool<TI,0>::Cnn *ConnectivityPool<TI,0>::create( BumpPointerPool &mem_pool, TI node_number ) {
    Cnn *res = mem_pool.create<Cnn>();
    res->prev_in_pool = last_in_pool;
    last_in_pool = res;

    res->node_number = node_number;

    return res;
}

// create ------------------------------------------------------------------------------------------------------------------------------------------------------------
template<class TI,int nvi>
void ConnectivityPool<TI,nvi>::write_to_stream( std::ostream &os ) const {
    next.write_to_stream( os );

    os << "\n  nvi " << nvi << ":";
    for( const Cnn *item = last_in_pool; item; item = item->prev_in_pool )
        item->write_to_stream( os << "\n    " );
}

template<class TI>
void ConnectivityPool<TI,0>::write_to_stream( std::ostream &/*os*/ ) const {
    //    os << "\n  nvi 0:";
    //    for( Cnn *item = last_in_pool; item; item = item->prev_in_pool )
    //        os << "\n    " << std::setw( 2 ) << item->tmp_num << ": " << item->node_number;
}


// apply_rec -----------------------------------------------------------------------------------------------------------------------------------------------------
template<class TI,int nvi> template<class F>
void ConnectivityPool<TI,nvi>::apply_rec( const F &f ) const {
    next.apply_rec( f );
    for( Cnn *item = last_in_pool; item; item = item->prev_in_pool )
        f( item );
}

template<class TI> template<class F>
void ConnectivityPool<TI,0>::apply_rec( const F &f ) const {
    for( Cnn *item = last_in_pool; item; item = item->prev_in_pool )
        f( item );
}

} // namespace sdot
} // namespace internal
} // namespace RecursivePolytop
