#include "variable_encode.h"
#include <iomanip>
#include <sstream>

namespace parex {

std::string variable_encode( const std::string &inp, bool disp_length ) {
    if ( disp_length ) {
        std::string out = variable_encode( inp, false );
        return std::to_string( out.size() ) + "_" + out;
    }

    std::ostringstream res;
    for( unsigned char c : inp ) {
        if ( ( c >= 'a' && c <= 'z' ) || ( c >= 'A' && c <= 'Z' ) || ( c >= '0' && c <= '9' ) )
            res << c;
        else if ( c == ' ' )
            res << "__";
        else
            res << "_" << std::hex << std::setfill( '0' ) << std::setw( 2 ) << unsigned( c );
    }

    return res.str();
}

} // namespace parex
