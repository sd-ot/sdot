#pragma once

#include <sstream>

template<class T,class ...Args>
std::string to_string( const T &val, const Args &...args ) {
    std::ostringstream ss;
    val.write_to_stream( ss, args... );
    return ss.str();
}

template<class T>
std::string to_string( const T &val ) {
    std::ostringstream ss;
    ss << val;
    return ss.str();
}

