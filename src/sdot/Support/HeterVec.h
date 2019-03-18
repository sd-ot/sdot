#pragma once

#include "TypeTransId.h"
#include "TypeTransId.h"
#include "Stream.h"
#include "Assert.h"
#include "Tail.h"
#include <vector>
#include <tuple>

/**
*/
template<class Tuple,class TypeTrans=TypeTransId>
struct HeterVec {
    using THead   = typename std::tuple_element<0,Tuple>::type;
    using TTail   = typename Tail<Tuple>::type;
    using NextVec = HeterVec<TTail,TypeTrans>;
    using CurType = typename TypeTrans::template Trans<THead>::T;

    template<class OthType>
    HeterVec &operator<<( OthType &&item ) {
        next << std::forward<OthType>( item );
        return *this;
    }

    HeterVec &operator<<( CurType &&item ) {
        list.emplace_back( std::forward<CurType>( item ) );
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
struct HeterVec<std::tuple<>,TypeTrans> {
    template<class OthType>
    HeterVec &operator<<( OthType &&item ) {
        ERROR( "type not permitted in this list" );
        return *this;
    }

    void write_to_stream( std::ostream &os, const char *sep = " ", const char *pre = 0 ) const {
    }

    std::size_t size() const {
        return 0;
    }

    template<class Func>
    void for_each( Func && ) const {
    }
};
