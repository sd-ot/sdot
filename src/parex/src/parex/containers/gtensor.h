#ifndef PAREX_gtensor_HEADER
#define PAREX_gtensor_HEADER

#include <asimd/allocators/AlignedAllocator.h>
#include "../N.h"
#include <array>

/**
   A "Generic" tensor that can be positionned on the main CPU memory or for instance on a GPU...

   Row major layout


*/
template<class T,std::size_t D,class Allocator=asimd::AlignedAllocator<T,64>>
class gtensor {
public:
    using                            I              = typename Allocator::size_type;
    using                            S              = std::array<I,D>;

    /**/                             gtensor        ( const gtensor & ) = delete;
    /**/                             gtensor        ( gtensor && ) = delete;
    /**/                             gtensor        ();

    /**/                            ~gtensor        ();

    gtensor&                         operator=      ( const gtensor & ) = delete;
    gtensor&                         operator=      ( gtensor && ) = delete;

    template<class... Args> void     resize         ( Allocator &allocator, Args&& ...args );


    void                             write_to_stream( std::ostream &os, const Allocator &allocator = {} ) const;
    template<class F> void           for_each_index ( F &&f ) const;
    template<class... Args> I        index          ( Args&& ...args ) const;
    I                                shape          ( I d ) const { return _size[ d ]; }
    S                                shape          () const { return _size; }
    S                                cpre           () const { return _cpre; }
    template<class... Args> T        at             ( const Allocator &allocator, Args&& ...args ) const;

    template<class... Z> const T*    data           ( Z&& ...args ) const { return _data + index( std::forward<Z>( args )... ); }
    template<class... Z> T*          data           ( Z&& ...args ) { return _data + index( std::forward<Z>( args )... ); }

    const T*                         data           () const { return _data; }
    T*                               data           () { return _data; }

private:
    template                         <class F,int d,class...Z>
    void                             _for_each_index( F &&f, N<d>, Z&& ...inds ) const;
    template                         <class F,class...Z>
    void                             _for_each_index( F &&f, N<D>, Z&& ...inds ) const;

    void                             _update_cpre   ();

    template<class... Args> I        _mul_cpre      ( I ind, I arg, Args&& ...args ) const { return _cpre[ ind ] * arg + _mul_cpre( ind + 1, std::forward<Args>( args )... ); }
    I                                _mul_cpre      ( I /*ind*/, I arg ) const { return arg; }
    I                                _mul_cpre      ( I /*ind*/ ) const { return 0; }

    S                                _size;         ///<
    S                                _rese;         ///<
    S                                _cpre;         ///<
    T*                               _data;         ///<
};

#include "gtensor.tcc"

#endif // PAREX_gtensor_HEADER
