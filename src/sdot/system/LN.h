#pragma once

#include "StaticGenericListOfNumbers.h"

/**
  List of known integers
*/
template<int... n>
struct LN;

template<int _head, int... _tail>
struct LN<_head,_tail...> : public StaticGenericListOfNumbers<LN<_head,_tail...>> {
    using Tail = LN<_tail...>;

    enum {
        has_only_zeros = _head == 0 && Tail::has_only_zeros,
        size           = 1 + Tail::size,
        head           = _head,
    };

    template<int nv>
    struct Append {
        using T = LN<_head,_tail...,nv>;
    };

    template<int nv>
    struct Prepend {
        using T = LN<nv,_head,_tail...>;
    };

    template<int off,int mul>
    struct AffineTrans {
        using A = typename Tail::template AffineTrans<off,mul>::T;
        using T = typename A::template Prepend< off + mul * head >::T;
    };

    template<int beg,int end>
    struct AllInRange {
        enum { value = head >= beg && head < end && Tail::template AllInRange<beg,end>::value };
    };

    template<int nv,int default_value=0>
    struct At {
        enum { value = Tail::template At<nv-1,default_value>::value };
    };
    template<int default_value>
    struct At<0,default_value> {
        enum { value = head };
    };

    template<int nv,int dummy=0>
    struct WithoutItemNb {
        using F = typename Tail::template WithoutItemNb<nv-1>::T;
        using T = typename F::template Prepend<head>::T;
    };
    template<int dummy>
    struct WithoutItemNb<0,dummy> {
        using T = Tail;
    };

    void write_to_stream( std::ostream &os, const char *sp = "" ) const {
        tail.write_to_stream( os << sp << head + 0, " " );
    }

    template<int index> constexpr
    static int at( N<index> ) {
        return At<index>::value;
    }

    template<int index> constexpr
    static int at() {
        return At<index>::value;
    }

    template<class TF>
    static void for_each( const TF &f ) {
        f( N<head>() );
        Tail::for_each( f );
    }

    template<class TF>
    static void for_each_with_cpt( const TF &f ) {
        _for_each_with_cpt( f, N<0>() );
    }

    template<class TF,int cpt>
    static void _for_each_with_cpt( const TF &f, N<cpt> nc ) {
        f( N<head>(), nc );
        Tail::_for_each_with_cpt( f, N<cpt+1>() );
    }

    static constexpr int first_nz_value( int default_value ) {
        return head != 0 ? head : Tail::first_nz_value( default_value );
    }

    static constexpr bool all_positive_of_null() {
        return head >= 0 && Tail::all_positive_of_null();
    }

    static constexpr bool all_negative_of_null() {
        return head <= 0 && Tail::all_negative_of_null();
    }

    template<int nv>
    LN<_head,_tail...,nv> append() {
        return {};
    }

    template<int nv>
    LN<_head,_tail...,nv> append( N<nv> ) {
        return {};
    }

    template<int nv>
    LN<nv,_head,_tail...> prepend() {
        return {};
    }

    template<int nv>
    LN<nv,_head,_tail...> prepend( N<nv> ) {
        return {};
    }

    // func( N<val> ) -> N<new_val>
    template<class Func>
    auto map( Func &&func ) const {
        return tail.map( func ).prepend( func( N<head>() ) );
    }

    static constexpr
    std::size_t offset_in_cube( std::size_t nb_items_per_axis, std::size_t off_beg = 0, std::size_t mul = 1 ) {
        return mul * ( head - off_beg ) + Tail::offset_in_cube( nb_items_per_axis, off_beg, mul * nb_items_per_axis );
    }

    template<class I,std::size_t s> static constexpr
    std::size_t offset_in_rect( const std::array<I,s> &rect_size, std::size_t mul = 1, unsigned d = 0 ) {
        return head * mul + Tail::offset_in_cube( rect_size, mul * rect_size[ d ], d + 1 );
    }

    auto operator-() const {
        return map( []( auto v ) { return -v; } );
    }

    Tail tail;
};

template<>
struct LN<> : public StaticGenericListOfNumbers<LN<>> {
    enum {
        has_only_zeros = true,
        size           = 0
    };

    template<int nv,int default_value=0>
    struct At {
        enum { value = default_value };
    };

    template<int nv>
    struct Append {
        using T = LN<nv>;
    };

    template<int nv>
    struct Prepend {
        using T = LN<nv>;
    };

    template<int off,int mul>
    struct AffineTrans {
        using T = LN<>;
    };

    template<int beg,int end>
    struct AllInRange {
        enum { value = true };
    };

    void write_to_stream( std::ostream &os, const char *sp = "" ) const {
    }

    template<class TF>
    static void for_each( const TF &f ) {
    }

    template<class TF>
    static void for_each_with_cpt( const TF &f ) {
    }

    template<class TF,int cpt>
    static void _for_each_with_cpt( const TF &f, N<cpt> nc ) {
    }

    static constexpr int first_nz_value( int default_value ) {
        return default_value;
    }

    static constexpr bool all_positive_of_null() {
        return true;
    }

    template<int nv>
    LN<nv> append() {
        return {};
    }

    template<int nv>
    LN<nv> append( N<nv> ) {
        return {};
    }

    template<int nv>
    LN<nv> prepend() {
        return {};
    }

    template<int nv>
    LN<nv> prepend( N<nv> ) {
        return {};
    }

    template<class Func>
    auto map( Func &&func ) const {
        return LN<>();
    }

    static constexpr bool all_negative_of_null() {
        return true;
    }

    static constexpr
    std::size_t offset_in_cube( std::size_t nb_items_per_axis, std::size_t off_beg = 0, std::size_t mul = 1 ) {
        return 0;
    }

    template<class I,std::size_t s> static constexpr
    std::size_t offset_in_rect( const std::array<I,s> &rect_size, std::size_t mul = 1, unsigned d = 0 ) {
        return 0;
    }

    auto operator-() const {
        return LN<>();
    }
};

template<int val,int size>
auto make_LN( N<val>, N<size> ) {
    return make_LN( N<val>(), N<size-1>() ).template prepend<val>();
}

template<int val>
auto make_LN( N<val>, N<0> ) {
    return LN<>();
}
