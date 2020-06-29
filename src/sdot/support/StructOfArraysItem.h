#pragma once

#include <ostream>
#include <vector>
#include <array>
#include <tuple>
#include "S.h"

/**
*/
template<class T,class TI,class Enable=void>
struct StructOfArraysItem {
    template<class F> void for_each_ptr( const F &f ) { f( data, S<T>() ); }
    void init( const TI *&/*vector_sizes*/ ) {}
    void  write_to_stream( std::ostream &os, TI ind, TI &cpt ) const { if ( cpt++ ) os << ","; os << data[ ind ]; }

    const T &operator[]( TI ind ) const { return data[ ind ]; }
    T &operator[]( TI ind ) { return data[ ind ]; }

    T   *data;
};

template<class TI>
struct StructOfArraysItem<std::tuple<>,TI> {
    template<class F> void for_each_ptr( const F &/*f*/ ) {}
    void init( const TI *&/*vector_sizes*/ ) {}
    void write_to_stream( std::ostream &/*os*/, TI /*ind*/, TI /*cpt*/ ) const {}
};

template<class Head,class... Tail,class TI>
struct StructOfArraysItem<std::tuple<Head,Tail...>,TI> {
    using Next = StructOfArraysItem<std::tuple<Tail...>,TI>;
    using Data = StructOfArraysItem<typename Head::T,TI>;

    template<class F> void for_each_ptr( const F &f ) { data.for_each_ptr( f ); next.for_each_ptr( f ); }
    void init ( const TI *&vector_sizes ) { data.init( vector_sizes ); next.init( vector_sizes ); }
    void write_to_stream( std::ostream &os, TI ind, TI &cpt ) const { data.write_to_stream( os, ind, cpt ); next.write_to_stream( os, ind, cpt ); }

    template<class A> const auto &operator[]( A a ) const { return next[ a ]; }
    template<class A> auto &operator[]( A a ) { return next[ a ]; }
    const auto &operator[]( Head ) const { return data; }
    auto &operator[]( Head ) { return data; }

    Data  data;
    Next  next;
};

template<class T,class TI>
struct StructOfArraysItem<std::vector<T>,TI> {
    using Item = StructOfArraysItem<T,TI>;
    using Data = std::vector<Item>;

    template<class F> void for_each_ptr( const F &f ) { for( Item &item : data ) item.for_each_ptr( f ); }
    void init( const TI *&vector_sizes ) { data.resize( *( vector_sizes++ ) ); for( Item &item : data ) item.init( vector_sizes ); }
    void write_to_stream( std::ostream &os, TI ind, TI &cpt ) const { if ( cpt++ ) os << ","; os << "["; TI ncpt = 0; for( const Item &item : data ) item.write_to_stream( os, ind, ncpt ); os << "]";  }

    TI size() const { return data.size(); }

    const Item &operator[]( TI ind ) const { return data[ ind ]; }
    Item &operator[]( TI ind ) { return data[ ind ]; }

    Data  data;
};

template<class T,std::size_t n,class TI>
struct StructOfArraysItem<std::array<T,n>,TI> {
    using Item = StructOfArraysItem<T,TI>;
    using Data = std::array<Item,n>;

    template<class F> void for_each_ptr( const F &f ) { for( Item &item : data ) item.for_each_ptr( f ); }
    void  init( const TI *&vector_sizes ) { for( Item &item : data ) item.init( vector_sizes ); }
    void  write_to_stream( std::ostream &os, TI ind, TI &cpt ) const { if ( cpt++ ) os << ","; os << "["; TI ncpt = 0; for( const Item &item : data ) item.write_to_stream( os, ind, ncpt ); os << "]";  }

    TI    size() const { return data.size(); }

    const Item &operator[]( TI ind ) const { return data[ ind ]; }
    Item &operator[]( TI ind ) { return data[ ind ]; }

    Data  data;
};
