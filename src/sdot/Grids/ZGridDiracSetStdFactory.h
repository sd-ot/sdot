#pragma once

#include "ZGridDiracSetFactory.h"
#include "ZGridDiracSetStd.h"

namespace sdot {

/**
*/
template<class Arch,class T,class S,int dim,class ContentByDirac>
class ZGridDiracSetStdFactory : public ZGridDiracSetFactory<T,S> {
public:
    virtual ZGridDiracSet<T,S> *New( ST size ) const override { return ZGridDiracSetStd<Arch,T,S,dim,ContentByDirac>::New( size ); }
};

}
