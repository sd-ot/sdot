#ifndef ALIGNED_VEC_SET_H
#define ALIGNED_VEC_SET_H

#include "StructOfArraysItem.h"
#include "AlignedAllocator.h"
#include "CpuArch.h"

/**
  struct Pos { using T = std::vector<std::array<float,3>>; };
  struct Ids { using T = std::vector<float>; };

  AlignedVecSet<std::tuple<Pos,Ids>> v( {  } );
*/
template<class Attributes,class Arch=CpuArch::Native,class TI=std::size_t>
struct StructOfArrays {
    using                   Items          = StructOfArraysItem<Attributes,TI>;

    /**/                    StructOfArrays ( const std::vector<TI> &vector_sizes = {}, TI rese = 1024 );
    /**/                    StructOfArrays ( StructOfArrays &&that );
    /**/                   ~StructOfArrays ();

    void                    write_to_stream( std::ostream &os ) const;
    template<class F> void  for_each_ptr   ( const F &f );
    template<class A> auto &operator[]     ( A a ) const { return data[ a ]; }
    template<class A> auto &operator[]     ( A a ) { return data[ a ]; }

    void                    reserve        ( TI new_rese, TI old_size );
    void                    reserve        ( TI new_rese );
    void                    clear          ();

    Items                   data;
    TI                      rese;          ///<
    TI                      size;          ///<
};

#include "StructOfArrays.tcc"

#endif // ALIGNED_VEC_SET_H
