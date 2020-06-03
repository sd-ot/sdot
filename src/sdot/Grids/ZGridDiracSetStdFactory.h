#pragma once

#include "ZGridDiracSetFactory.h"
#include "ZGridDiracSetStd.h"

namespace sdot {

/**
*/
template<class ContentByDirac>
class ZGridDiracSetStdFactory : public ZGridDiracSetFactory {
public:
    virtual ZGridDiracSet *New( ST size ) const override { return ZGridDiracSetStd<ContentByDirac>::New( size ); }
};

}
