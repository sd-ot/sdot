#pragma once

#include <array>
#include <tuple>
#include "Vec.h"
#include "S.h"

namespace sdot {

/**
  Specialization for a simple scalar
*/
template<class T,class TI,class Arch>
struct StructOfArraysItem {
    void                          set_vector_sizes( const TI *&/*vector_sizes*/ ) {}
    void                          write_to_stream ( std::ostream &os, TI ind, TI &cpt ) const { if ( cpt++ ) os << ","; os << vec[ ind ]; }
    template<class F> void        for_each_vec    ( const F &f ) { f( vec ); }

    T                             operator[]      ( TI ind ) const { return vec[ ind ]; }

    const T*                      ptr             () const { return vec.data(); }
    T*                            ptr             () { return vec.data(); }

    Vec<T,Arch>                   vec;
};

/**
  Specialization for a void aggregate
*/
template<class TI,class Arch>
struct StructOfArraysItem<std::tuple<>,TI,Arch> {
    void                          write_to_stream ( std::ostream &/*os*/, TI /*ind*/, TI /*cpt*/ ) const {}
    template<class F> void        for_each_vec    ( const F &/*f*/ ) {}
    void                          set_vector_sizes( const TI *&/*vector_sizes*/ ) {}
};

/**
  Specialization for an aggregate
*/
template<class Head,class... Tail,class TI,class Arch>
struct StructOfArraysItem<std::tuple<Head,Tail...>,TI,Arch> {
    using                         Next            = StructOfArraysItem<std::tuple<Tail...>,TI,Arch>;
    using                         Data            = StructOfArraysItem<typename Head::T,TI,Arch>;

    void                          set_vector_sizes( const TI *&vector_sizes ) { data.set_vector_sizes( vector_sizes ); next.set_vector_sizes( vector_sizes ); }
    void                          write_to_stream ( std::ostream &os, TI ind, TI &cpt ) const { data.write_to_stream( os, ind, cpt ); next.write_to_stream( os, ind, cpt ); }
    template<class F> void        for_each_vec    ( const F &f ) { data.for_each_vec( f ); next.for_each_vec( f ); }

    template<class A> const auto& operator[]      ( A a ) const { return next[ a ]; }
    template<class A> auto&       operator[]      ( A a ) { return next[ a ]; }
    const auto&                   operator[]      ( Head ) const { return data; }
    auto&                         operator[]      ( Head ) { return data; }

    Data                          data;
    Next                          next;
};

/**
  Specialization for a vector
*/
template<class T,class TI,class Arch>
struct StructOfArraysItem<std::vector<T>,TI,Arch> {
    using                         Item            = StructOfArraysItem<T,TI,Arch>;
    using                         Data            = std::vector<Item>;

    void                          set_vector_sizes( const TI *&vector_sizes ) { data.resize( *( vector_sizes++ ) ); for( Item &item : data ) item.set_vector_sizes( vector_sizes ); }
    void                          write_to_stream ( std::ostream &os, TI ind, TI &cpt ) const { if ( cpt++ ) os << ","; os << "["; TI ncpt = 0; for( const Item &item : data ) item.write_to_stream( os, ind, ncpt ); os << "]";  }
    template<class F> void        for_each_vec    ( const F &f ) { for( Item &item : data ) item.for_each_vec( f ); }

    TI                            size            () const { return data.size(); }

    const Item &                  operator[]      ( TI ind ) const { return data[ ind ]; }
    Item&                         operator[]      ( TI ind ) { return data[ ind ]; }

    Data                          data;
};

/**
  Specialization for an array
*/
template<class T,std::size_t n,class TI,class Arch>
struct StructOfArraysItem<std::array<T,n>,TI,Arch> {
    using                         Item            = StructOfArraysItem<T,TI,Arch>;
    using                         Data            = std::array<Item,n>;

    void                          set_vector_sizes( const TI *&vector_sizes ) { for( Item &item : data ) item.set_vector_sizes( vector_sizes ); }
    void                          write_to_stream ( std::ostream &os, TI ind, TI &cpt ) const { if ( cpt++ ) os << ","; os << "["; TI ncpt = 0; for( const Item &item : data ) item.write_to_stream( os, ind, ncpt ); os << "]";  }
    template<class F> void        for_each_vec    ( const F &f ) { for( Item &item : data ) item.for_each_vec( f ); }

    TI                            size            () const { return data.size(); }

    const Item&                   operator[]      ( TI ind ) const { return data[ ind ]; }
    Item&                         operator[]      ( TI ind ) { return data[ ind ]; }

    Data                          data;
};

} // namespace sdot
