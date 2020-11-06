#ifndef SDOT_VEC_H
#define SDOT_VEC_H

#include "AlignedAllocator.h"
#include <vector>

/**
  Aligned Vector
*/
template<class T,class Arch=CpuArch::Native>
class Vec<T,Arch> : public std::vector<T,AlignedAllocator<T,Arch>> {
public:
    using std::vector<T,AlignedAllocator<T,Arch>>::vector;
};

#endif // SDOT_VEC_H
