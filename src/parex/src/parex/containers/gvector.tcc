#include "../utility/TODO.h"
#include "gvector.h"

namespace parex {

template<class T,class A>
gvector<T,A>::gvector( A *allocator, I size, I rese, T *data, bool own ) : P( allocator, S{ size }, S{ rese }, data, own ) {
}

template<class T,class A>
gvector<T,A>::gvector( A *allocator, I size, T *data, bool own ) : gvector( allocator, size, size, data, own ) {
}

template<class T,class A> template<class U>
gvector<T,A>::gvector( A *allocator, std::initializer_list<U> &&l ) : P( allocator, std::move( l ) ) {
}

template<class T,class A>
gvector<T,A>::gvector( gvector &&that ) : P( std::move( that ) ) {
}

template<class T,class A>
gvector<T,A>::gvector( const gvector &that ) : P( that ) {
}

template<class T,class A>
gvector<T,A>::gvector() {
}

template<class T,class A>
void gvector<T,A>::resize( I new_size ) {
    P::resize( new_size );
}

} // namespace parex
