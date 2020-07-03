#pragma once

#include <functional>
#include <vector>

template<class TI>
bool for_each_permutation_cont( const std::vector<TI> &v, const std::function<bool(const std::vector<TI> &)> &f, const std::vector<TI> &b = {} ) {
    if ( v.empty() )
        return f( b );

    for( TI i = 0; i < v.size(); ++i ) {
        std::vector<TI> nb = b, nv = v;
        nv.erase( nv.begin() + i );
        nb.push_back( v[ i ] );
        if ( ! for_each_permutation_cont( nv, f, nb ) )
            return false;
    }

    return true;
}

template<class TI>
void for_each_permutation( const std::vector<TI> &v, const std::function<void(const std::vector<TI> &)> &f, const std::vector<TI> &b = {} ) {
    if ( v.empty() )
        return f( b );

    for( TI i = 0; i < v.size(); ++i ) {
        std::vector<TI> nb = b, nv = v;
        nv.erase( nv.begin() + i );
        nb.push_back( v[ i ] );
        for_each_permutation( nv, f, nb );
    }
}
