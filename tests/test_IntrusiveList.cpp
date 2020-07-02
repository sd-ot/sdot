#include "../src/sdot/support/IntrusiveList.h"
#include "../src/sdot/support/P.h"
#include <vector>

struct Smurf {
    void   write_to_stream( std::ostream &os ) const { os << val; }
    Smurf *next;
    int    val;
};

int main() {
    using TI = std::size_t;

    std::vector<Smurf> s( 20 );
    for( TI i = 0; i < s.size(); ++i )
        s[ i ].val = i;

    IntrusiveList<Smurf> l;
    for( TI i = s.size(); i--; )
        l.push_front( s.data() + i );

    P( l );

    IntrusiveList<Smurf> m;
    l.move_to_if( m, []( const Smurf &s ) { return s.val % 2 == 0; } );
    P( l );
    P( m );
}
