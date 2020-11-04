#pragma once

#include <vector>
#include <string>

namespace parex {

/**
*/
class Data {
public:
    /**/        Data     ( std::string type = {}, void *ptr = nullptr ) : type( type ), ptr( ptr ) {}
    /**/        Data     ( const Data &that );
    /**/        Data     ( Data &&that );

    Data&       operator=( const Data &that );
    Data&       operator=( Data &&that );

    std::string type;
    void*       ptr;
};

inline Data data_from_value( std::int32_t  value ) { return Data{ "std::int32_t" , new std::int32_t( value ) }; }
inline Data data_from_value( std::ostream* value ) { return Data{ "std::ostream*", value }; }

} // namespace parex
