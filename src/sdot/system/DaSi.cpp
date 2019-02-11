#include "DaSi.h"

namespace Hpipe {

void DaSi::write_to_stream( std::ostream &os ) const {
    os.write( data, size );
}

PT DaSi::find( char c ) const {
    for( PT i = 0; ; ++i )
        if ( i == size or data[ i ] == c )
            return i;
}

PT DaSi::rfind( char c ) const {
    for( PT i = size; ; ) {
        if ( not i )
            return size;
        if ( data[ --i ] == c )
            return i;
    }
}

} // namespace Hpipe

