#ifndef PAREX_gvector_HEADER
#define PAREX_gvector_HEADER

#include "gtensor.h"

namespace parex {

/**
   A "generic" vector with memory handled with a parex allocator (i.e. that can be on a GPU, a CPU, ...)
*/
template<class T,class Allocator=CpuAllocator>
class gvector : public gtensor<T,1,Allocator> {
public:
    using                            P              = gtensor<T,1,Allocator>;
    using                            I              = typename P::I;
    using                            S              = typename P::S;

    /**/                             gvector        ( Allocator *allocator, I size = 0, T *data = nullptr, bool own = true ); ///< data is NOT copied but taken as is for the content
    /**/                             gvector        ( Allocator *allocator, I size, I rese, T *data = nullptr, bool own = true ); ///< data is NOT copied but taken as is for the content
    /**/                             gvector        ( const gvector & ); ///< we need the allocator to make a copy
    /**/                             gvector        ( gvector && );

    gvector&                         operator=      ( const gvector & ) = delete;
    gvector&                         operator=      ( gvector && ) = delete;

    void                             resize         ( I new_size );
    I                                size           () const { return this->shape( 0 ); }
};

} // namespace parex

#include "gvector.tcc"

#endif // PAREX_gvector_HEADER
