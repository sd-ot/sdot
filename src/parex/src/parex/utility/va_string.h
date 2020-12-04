#pragma once

#include "generic_ostream_output.h"
#include <sstream>
#include <string>

namespace parex {

inline std::string va_string( const std::string &str ) {
    return str;
}

template<class Head,class ...Tail>
std::string va_string( const std::string &str, const Head &head, const Tail &...tail ) {
    std::string::size_type pos = str.find( "{}" );
    if ( pos == std::string::npos )
        return str;
    std::ostringstream ss;
    ss << str.substr( 0, pos ) << head << str.substr( pos + 2 );
    return va_string( ss.str(), tail... );
}

} // namespace parex
