#ifndef SDOT_StructOfArrays_H
#define SDOT_StructOfArrays_H

#include "StructOfArraysItem.h"
#include "../support/Vec.h"

namespace sdot {

/**
  struct Pos { using T = std::vector<std::array<float,3>>; };
  struct Ids { using T = std::vector<float>; };

  AlignedVecSet<std::tuple<Pos,Ids>> v( {  } );
*/
template<class Attributes,class Arch=MachineArch::Native,class TI=std::size_t>
struct StructOfArrays {
    using                   Items          = StructOfArraysItem<Attributes,TI,Arch>;

    /**/                    StructOfArrays ( const std::vector<TI> &vector_sizes = {}, TI _rese = 1024 );

    void                    write_to_stream( std::ostream &os ) const;
    template<class F> void  for_each_vec   ( const F &f );
    template<class A> auto &operator[]     ( A a ) const { return _data[ a ]; }
    template<class A> auto &operator[]     ( A a ) { return _data[ a ]; }
    TI                      size           () const;

    void                    resize_wo_check( TI new_size );
    void                    reserve        ( TI new_rese, TI old_size );
    void                    reserve        ( TI new_rese );
    void                    resize         ( TI new_size );
    void                    clear          ();

private:
    Items                   _data;          ///<
    TI                      _rese;          ///<
    TI                      _size;          ///<
};

} // namespace sdot

#include "StructOfArrays.tcc"

#endif // SDOT_StructOfArrays_H
