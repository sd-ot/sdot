#include "serialized.h"
#include <cstring>

namespace parex {

std::string serialized_bin( std::string type, const void *data, std::size_t size ) {
    std::string res;
    res.resize( type.size() + 1 + size );
    char *p = const_cast<char *>( res.data() );

    std::memcpy( p                  , type.data(), type.size() );
    std::memcpy( p + type.size()    , ";"        , 1 );
    std::memcpy( p + type.size() + 1, data       , size );

    return res;
}

} // namespace parex
