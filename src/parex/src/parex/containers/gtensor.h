#ifndef PAREX_gtensor_HEADER
#define PAREX_gtensor_HEADER

#include "CpuAllocator.h"
#include "../utility/N.h"
#include <ostream>
#include <array>

namespace parex {

/**
   A "generic" tensor with memory handled with a parex allocator (i.e. that can be on a GPU, a CPU, ...)

   Row major layout (no choice)
*/
template<class T,int D,class Allocator=CpuAllocator>
class gtensor {
public:
    using                            I              = typename Allocator::I;
    using                            S              = std::array<I,D>;

    /**/                             gtensor        ( Allocator *allocator, S size = _null_S(), T *data = nullptr, bool own = true ); ///< data is NOT copied but taken as is for the content
    /**/                             gtensor        ( Allocator *allocator, S size, S rese, T *data = nullptr, bool own = true ); ///< data is NOT copied but taken as is for the content
    /**/                             gtensor        ( const gtensor & ); ///< we need the allocator to make a copy
    /**/                             gtensor        ( gtensor && );

    /**/                            ~gtensor        ();

    gtensor&                         operator=      ( const gtensor & ) = delete;
    gtensor&                         operator=      ( gtensor && ) = delete;

    template<class... Args> void     resize         ( Args&& ...args );

    void                             write_to_stream( std::ostream &os ) const;
    template<class F> void           for_each_index ( F &&f ) const;
    template<class... Args> I        index          ( Args&& ...args ) const;
    I                                shape          ( I d ) const { return _size[ d ]; }
    S                                shape          () const { return _size; }
    S                                cpre           () const { return _cprs; }
    template<class... Args> T        at             ( Args&& ...args ) const;

    template<class... Z> const T*    data           ( Z&& ...args ) const { return _data + index( std::forward<Z>( args )... ); }
    template<class... Z> T*          data           ( Z&& ...args ) { return _data + index( std::forward<Z>( args )... ); }

    const T*                         data           () const { return _data; }
    T*                               data           () { return _data; }

private:
    template                         <class F,int d,class...Z>
    void                             _for_each_index( F &&f, N<d>, Z&& ...inds ) const;
    template                         <class F,class...Z>
    void                             _for_each_index( F &&f, N<D>, Z&& ...inds ) const;

    void                             _update_rese   ();
    void                             _update_cprs   ();
    void                             _allocate      ();
    static S                         _null_S        ();
    void                             _clear         ();

    template<class... Args> I        _mul_cprs      ( I ind, I arg, Args&& ...args ) const { return _cprs[ ind ] * arg + _mul_cprs( ind + 1, std::forward<Args>( args )... ); }
    I                                _mul_cprs      ( I /*ind*/, I arg ) const { return arg; }
    I                                _mul_cprs      ( I /*ind*/ ) const { return 0; }

    Allocator*                       _allocator;    ///<
    S                                _size;         ///<
    S                                _rese;         ///<
    S                                _cprs;         ///< cumulative product of reservation sizes
    T*                               _data;         ///<
    bool                             _own;          ///< if data is owned
};

} // namespace parex

#include "gtensor.tcc"

#endif // PAREX_gtensor_HEADER
