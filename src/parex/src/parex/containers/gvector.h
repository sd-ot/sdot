#ifndef PAREX_gvector_HEADER
#define PAREX_gvector_HEADER

#include "gtensor.h"

namespace parex {

/**
   A "generic" vector with memory handled with a parex allocator (i.e. that can be on a GPU, a CPU, ...)
*/
template<class T,class Allocator=BasicCpuAllocator>
class gvector : public gtensor<T,1,Allocator> {
public:
    using                            PA       = gtensor<T,1,Allocator>;
    using                            I        = typename PA::I;
    using                            S        = typename PA::S;

    /**/                             gvector  ( Allocator *allocator, I size, I rese, T *data, bool own = true ); ///< data is NOT copied but taken as is for the content
    /**/                             gvector  ( Allocator *allocator, I size, T *data, bool own = true ); ///< data is NOT copied but taken as is for the content
    template<class U>                gvector  ( Allocator *allocator, std::initializer_list<U> &&l ); ///< data is NOT copied but taken as is for the content
    /**/                             gvector  ( const gvector & ); ///< we need the allocator to make a copy
    /**/                             gvector  ( gvector && );
    /**/                             gvector  ();

    gvector&                         operator=( const gvector & ) = delete;
    gvector&                         operator=( gvector && ) = delete;

    void                             resize   ( I new_size );
    I                                size     () const { return this->shape( 0 ); }
};

template<class T,class A>
struct TypeInfo<gvector<T,A>> {
    static std::string name() {
        return "parex::gvector<" + TypeInfo<T>::name() + "," + TypeInfo<A>::name() + ">";
    }
};

template<class B,class T,class A>
gvector<T,B> *new_copy_in( B &new_allocator, const gvector<T,A> &value );

} // namespace parex

#include "gvector.tcc"

#endif // PAREX_gvector_HEADER
