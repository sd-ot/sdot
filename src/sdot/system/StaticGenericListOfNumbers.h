#pragma once

#include "Stream.h"
#include "N.h"

template<class PT>
struct StaticGenericListOfNumbers {
    template<class T>
    operator std::vector<T>() const {
        const PT &pt = reinterpret_cast<const PT &>( *this );
        std::vector<T> res( pt.size );

        pt.for_each_with_cpt( [&]( auto n_val, auto cpt ) {
            res[ cpt.val ] = n_val.val;
        } );
        return res;
    }

    template<class T,int size>
    operator std::array<T,size>() const {
        const PT &pt = reinterpret_cast<const PT &>( *this );
        std::array<T,size> res;

        pt.for_each_with_cpt( [&]( auto n_val, auto cpt ) {
            res[ cpt.val ] = n_val.val;
        } );
        return res;
    }

};

