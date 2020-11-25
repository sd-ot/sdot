#include "url_encode.h"
#include <vector>

using namespace std;

namespace parex {

void hexchar( unsigned char c, unsigned char &hex1, unsigned char &hex2 ) {
    hex1 = c / 16;
    hex2 = c % 16;
    hex1 += hex1 <= 9 ? '0' : 'a' - 10;
    hex2 += hex2 <= 9 ? '0' : 'a' - 10;
}

string url_encode( string s ) {
    vector<char> v;
    v.reserve( s.size() );

    const char *str = s.c_str();
    for( size_t i = 0, l = s.size(); i < l; i++ ) {
        char c = str[i];
        if ( ( c >= '0' && c <= '9' ) ||
             ( c >= 'a' && c <= 'z' ) ||
             ( c >= 'A' && c <= 'Z' ) ||
             c == '-' || c == '_' || c == '.' ) {
            v.push_back( c );
        } else if ( c == ' ' ) {
            v.push_back( '+' );
        } else {
            v.push_back( '%' );
            unsigned char d1, d2;
            hexchar( c, d1, d2 );
            v.push_back( d1 );
            v.push_back( d2 );
        }
    }

    return string( v.cbegin(), v.cend() );
}


} // namespace parex
