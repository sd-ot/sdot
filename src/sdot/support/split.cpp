#include "split.h"

std::vector<std::string> split( std::string str, std::string sep ) {
    std::size_t o = 0;
    std::vector<std::string> res;
    for( std::size_t i = 0; i + sep.size() < str.size(); ++i ) {
        if ( str.substr( i, sep.size() ) == sep ) {
            res.push_back( str.substr( o, i - o ) );
            o = i + sep.size();
        }
    }
    if ( std::size_t d = str.size() - o )
        res.push_back( str.substr( o, d ) );
    return res;
}
