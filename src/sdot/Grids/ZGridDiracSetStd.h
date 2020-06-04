#pragma once

#include "../support/BumpPointerPool.h"
#include "../support/SimdAlig.h"
#include "ZGridDiracSet.h"

namespace sdot {

/**
*/
template<class Arch,class T,class S,int dim,class ItemPerDirac>
class ZGridDiracSetStd : public ZGridDiracSet<T,S> {
public:
    static constexpr S       alf              = SimdAlig<Arch,T>::value;
    static constexpr S       alb              = alf * sizeof( T );

    virtual                 ~ZGridDiracSetStd ();

    static S                 nb_diracs_for_mem( std::size_t mem );
    static ZGridDiracSetStd* New              ( BumpPointerPool &pool, S size );

    virtual void             write_to_stream  ( std::ostream &os, const std::string &sp = {} ) const override;
    virtual void             get_base_data    ( T **coords, T *&weights, S *&ids ) override;
    virtual S                size             () override;

    T*                       coords           ( int d );
    S*                       ids              ();

    S                        _size;
    S                        _rese;
    // ... followed by coords, weight and id data
};

}

#include "ZGridDiracSetStd.tcc"
