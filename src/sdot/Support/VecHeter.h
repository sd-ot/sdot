#pragma once

#include "TypeTransId.h"
#include "Stream.h"
#include "Assert.h"
#include "Tail.h"
#include <vector>
#include <tuple>

/**
*/
template<class Tuple,class TypeTrans=TypeTransId>
struct VecHeter {
    using THead   = typename std::tuple_element<0,Tuple>::type;
    using TTail   = typename Tail<Tuple>::type;
    using NextVec = VecHeter<TTail>;
    using CurType = typename TypeTrans::template Trans<THead>::T;

    VecHeter &operator<<( const CurType &item ) {
        list.emplace_back( item );
        return *this;
    }

    template<class OthType>
    VecHeter &operator<<( const OthType &item ) {
        next << item;
        return *this;
    }

    template<class OthType>
    VecHeter &operator>>( const OthType &item ) {
        next >> item;
        return *this;
    }

    VecHeter &operator>>( const CurType &item ) {
        for( std::size_t i = 0; i < list.size(); ++i ) {
            if ( list[ i ] == item ) {
                list.erase( list.begin() + i );
                return *this;
            }
        }
        next >> item;
        return *this;
    }

    void write_to_stream( std::ostream &os, const char *sep = " ", const char *pre = 0 ) const {
        for( const CurType &val : list ) {
            if ( pre ) os << pre;
            pre = sep;
            os << val;
        }
        next.write_to_stream( os, sep, pre );
    }

    std::size_t size() const {
        return list.size() + next.size();
    }

    bool empty() const {
        return list.empty() && next.empty();
    }

    template<class OthType>
    bool contains( const OthType &item ) const {
        return next.contains( item );
    }

    bool contains( const CurType &item ) const {
        for( const CurType &v : list )
            if ( v == item )
                return true;
        return next.contains( item );
    }

    template<class Func>
    void for_each( Func &&func ) const {
        for( const CurType &val : list )
            func( val );
        next.for_each( std::forward<Func>( func ) );
    }

    std::vector<CurType> list;
    NextVec              next;
};

// specialization for a void list of type
template<class TypeTrans>
struct VecHeter<std::tuple<>,TypeTrans> {
    template<class OthType>
    VecHeter &operator<<( const OthType &item ) {
        P( typeid( OthType ).name() );
        ERROR( "type not permitted in this list" );
        return *this;
    }

    template<class OthType>
    VecHeter &operator>>( const OthType &item ) {
        return *this;
    }

    void write_to_stream( std::ostream &os, const char *sep = " ", const char *pre = 0 ) const {
    }

    std::size_t size() const {
        return 0;
    }

    bool empty() const {
        return true;
    }

    template<class OthType>
    bool contains( const OthType &item ) const {
        return false;
    }

    template<class Func>
    void for_each( Func && ) const {
    }
};
