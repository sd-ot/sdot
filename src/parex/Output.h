#pragma once

#include <utility>
#include <string>

namespace parex {

/**
*/
class Output {
public:
    /**/        Output   ( const std::string &type, void *data, bool own = true ) : type( type ), data( data ), own( own ) {}
    /**/        Output   ( const Output &that ) = delete;
    /**/        Output   ( Output &&that ) : type( std::exchange( that.type, {} ) ), data( std::exchange( that.data, nullptr ) ), own( std::exchange( that.own, false ) ) {}
    /**/        Output   () : data( nullptr ), own( false ) {}

    Output&     operator=( const Output &that ) = delete;
    Output&     operator=( Output &&that ) { type = std::exchange( that.type, {} ); data = std::exchange( that.data, nullptr ); own = std::exchange( that.own, false ); return *this; }

    void        destroy  ();

    std::string type;
    void*       data;
    bool        own;
};

} // namespace parex
