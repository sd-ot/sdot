#pragma once

#include "../support/SimdAlig.h"
#include "ZGridDiracSet.h"

namespace sdot {

/**
*/
template<class ItemPerDirac>
class ZGridDiracSetStd : public ZGridDiracSet {
public:
    static constexpr ST      alf             = SimdAlig<ARCH,TF>::value;
    static constexpr ST      alb             = alf * sizeof( TF );

    virtual                 ~ZGridDiracSetStd();

    static ZGridDiracSetStd* New             ( ST size );

    virtual void             get_base_data   ( TF **coords, TF *&weights, ST *&ids ) override;
    virtual ST               size            () override;

    TF*                      coords          ( int dim );
    ST*                      ids             ();

    ST                       _size;
    ST                       _rese;
    // ... followed by coords, weight and id data
};

}

#include "ZGridDiracSetStd.tcc"
