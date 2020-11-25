#include "cstr_encode.h"

using namespace std;

namespace parex {

string cstr_encode( string s ) {
    string res;
    for( char c : s ) {
        switch ( c ) {
        case '\n': res += "\\n" ; break;
        case '\\': res += "\\\\"; break;
        case '\"': res += "\\\""; break;
        default: res += c;
        }
    }
    return res;
}


} // namespace parex
