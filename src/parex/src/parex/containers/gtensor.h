#ifndef PAREX_gtensor_HEADER
#define PAREX_gtensor_HEADER

#include "../resources/BasicCpuAllocator.h"
#include "../utility/N.h"
#include <ostream>
#include <array>

namespace parex {

/**
   A "generic" tensor with memory handled with a parex allocator (i.e. that can be on a GPU, a CPU, ...)

   Row major layout (no choice)
*/
template<class T,int D,class Allocator=BasicCpuAllocator>
class gtensor {
public:
    using                            I                        = typename Allocator::I;
    using                            S                        = std::array<I,D>;


    template<class U>                gtensor                  ( Allocator *allocator, std::initializer_list<std::initializer_list<std::initializer_list<U>>> &&l ); ///<
    template<class U>                gtensor                  ( Allocator *allocator, std::initializer_list<std::initializer_list<U>> &&l ); ///<
    template<class U>                gtensor                  ( Allocator *allocator, std::initializer_list<U> &&l ); ///<
    /**/                             gtensor                  ( Allocator *allocator ); ///<
    /**/                             gtensor                  ( const gtensor & ); ///< we need the allocator to make a copy
    /**/                             gtensor                  ( gtensor && );

    static gtensor                   from_data                ( Allocator *allocator, S shape, T *data, bool own = true ); ///< data is NOT copied but taken as is for the content
    static gtensor                   from_data                ( Allocator *allocator, S shape, S rese, T *data, bool own = true ); ///< data is NOT copied but taken as is for the content
    static gtensor                   empty                    ( Allocator *allocator, S shape, S rese ); ///< data is allocated inside the ctor
    static gtensor                   empty                    ( Allocator *allocator, S shape ); ///< data is allocated inside the ctor

    /**/                            ~gtensor                  ();

    gtensor&                         operator=                ( const gtensor & ) = delete;
    gtensor&                         operator=                ( gtensor && ) = delete;

    template<class... Args> void     resize                   ( Args&& ...args );

    template<class F> void           for_each_offset_and_index( F &&f ) const;
    template<class F,class P> void   for_each_oai_simd_mt     ( F &&f, P &thread_pool ) const; ///< f( I beg_offset, int num_thread, N<simd_size>, SimdVec<I> ind_0, ... )
    template<class F,class P> void   for_each_oajk_mt         ( F &&f, P &thread_pool ) const; ///< f( I beg_offset, int num_thread, I ind_0, ... )
    template<class F> void           for_each_index           ( F &&f ) const;

    void                             write_to_stream          ( std::ostream &os ) const;

    static void                      update_rese              ( S &rese, Allocator &allocator );
    Allocator*                       allocator                () const { return _allocator; }
    I                                nb_items                 () const { return _cprs[ 0 ]; }
    template<class... Args> I        offset                   ( Args&& ...args ) const;
    I                                shape                    ( I d ) const { return _shape[ d ]; }
    S                                shape                    () const { return _shape; }
    S                                rese                     () const { return _rese; }
    S                                cprs                     () const { return _cprs; }
    template<class... Args> T        at                       ( Args&& ...args ) const;

    template<class... Z> const T*    data                     ( Z&& ...args ) const { return _data + offset( std::forward<Z>( args )... ); }
    template<class... Z> T*          data                     ( Z&& ...args ) { return _data + offset( std::forward<Z>( args )... ); }

    const T*                         data                     () const { return _data; }
    T*                               data                     () { return _data; }

private:
    template                         <class F,int d,class...Z>
    void                             _for_each_offset_and_index( F &&f, N<d>, I off, Z&& ...inds ) const;
    template                         <class F,class...Z>
    void                             _for_each_offset_and_index( F &&f, N<D-1>, I off, Z&& ...inds ) const;

    template                         <class F,int d,class...Z>
    void                             _for_each_index          ( F &&f, N<d>, Z&& ...inds ) const;
    template                         <class F,class...Z>
    void                             _for_each_index          ( F &&f, N<D>, Z&& ...inds ) const;

    void                             _update_cprs             ();
    void                             _allocate                ();
    static S                         _null_S                  ();
    void                             _clear                   ();

    void                             _set_at                  ( I index, const T &value );
    T                                _get_at                  ( I index ) const;

    template<class... Args> I        _mul_cprs                ( I ind, I arg, Args&& ...args ) const { return _cprs[ ind ] * arg + _mul_cprs( ind + 1, std::forward<Args>( args )... ); }
    I                                _mul_cprs                ( I /*ind*/, I arg ) const { return arg; }
    I                                _mul_cprs                ( I /*ind*/ ) const { return 0; }

    Allocator*                       _allocator;              ///<
    S                                _shape;                  ///<
    S                                _rese;                   ///<
    S                                _cprs;                   ///< cumulative product of reservation sizes
    T*                               _data;                   ///<
    bool                             _own;                    ///< if data is owned
};

template<class T,int D,class A>
struct TypeInfo<gtensor<T,D,A>> {
    static std::string name() {
        return "parex::gtensor<" + TypeInfo<T>::name() + "," + std::to_string( D ) + "," + TypeInfo<A>::name() + ">";
    }
};

template<class B,class T,int D,class A>
gtensor<T,D,B> *new_copy_in( B &new_allocator, const gtensor<T,D,A> &value );

} // namespace parex

#include "gtensor.tcc"

#endif // PAREX_gtensor_HEADER
