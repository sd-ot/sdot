#include "CmString.h"
#include <iostream>

namespace Hpipe {

void CmString::write_to_stream( std::ostream &os ) const {
    os.write( (const char *)beg, end - beg );
    //    static const char *c = "0123456789abcdef";

    //    PT cpt = 0;
    //    for( const PI8 *i = beg; i < end; ++i )
    //        os << ( cpt++ ? " " : "" ) << c[ *i / 16 ] << c[ *i % 16 ];
}

} // namespace Hpipe
