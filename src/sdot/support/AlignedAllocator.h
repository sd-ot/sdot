#pragma once

#include <type_traits>
#include <utility>
#include <cstdlib>

struct AlignedAllocator {
    template<class T,std::size_t alig=1>
    static T *allocate( std::size_t rese ) {
        return new ( aligned_alloc( alignof( T ) * alig, sizeof( T ) * rese ) ) T[ rese ];
    }

    template<class T>
    static void free( T *ptr, std::size_t rese ) {
        if ( rese ) {
            if ( ! std::is_trivially_destructible<T>::value )
                while ( rese-- )
                    ptr[ rese ].~T();
            std::free( ptr );
        }
    }

    template<class T,std::size_t alig=1>
    static void reallocate( T *&ptr, std::size_t size, std::size_t rese, std::size_t new_rese ) {
        T *res = reinterpret_cast<T *>( aligned_alloc( alignof( T ) * alig, sizeof( T ) * new_rese ) );
        for( std::size_t i = 0; i < size; ++i )
            new ( res + i ) T( std::move( ptr[ i ] ) );
        for( std::size_t i = size; i < new_rese; ++i )
            new ( res + i ) T;

        if ( rese ) {
            if ( ! std::is_trivially_destructible<T>::value )
                while ( rese-- )
                    ptr[ rese ].~T();
            std::free( ptr );
        }

        ptr = res;
    }
};
