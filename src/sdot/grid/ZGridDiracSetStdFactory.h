#pragma once

#include "ZGridDiracSetFactory.h"
#include "ZGridDiracSetStd.h"

namespace sdot {

/**
*/
template<class Arch,class T,class S,int dim,class ContentByDirac>
class ZGridDiracSetStdFactory : public ZGridDiracSetFactory<T,S> {
public:
    virtual S                   nb_diracs_for_mem( std::size_t mem ) const { return ZGridDiracSetStd<Arch,T,S,dim,ContentByDirac>::nb_diracs_for_mem( mem ); }
    virtual ZGridDiracSet<T,S> *New              ( BumpPointerPool &pool, ST size ) const override { return ZGridDiracSetStd<Arch,T,S,dim,ContentByDirac>::New( pool, size ); }
};

}
