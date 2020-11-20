#include "PowerDiagram.h"
#include <parex/support/P.h>

using namespace parex;
using std::min;
using std::max;

namespace sdot {

PowerDiagram::PowerDiagram( int dim ) : dim( dim ) {
}

void PowerDiagram::add_diracs( const parex::Value &diracs ) {
    this->diracs.push_back( diracs.ref );
}

void PowerDiagram::for_each_cell( const std::function<void(const Value &)> &f ) {
    if ( diracs.empty() )
        return;

    // get min/max
    std::vector<TaskRef> min_maxs( diracs.size() );
    for( std::size_t i = 0; i < diracs.size(); ++i )
        min_maxs[ i ] = Task::call_r( "sdot/geometry/kernels/PowerDiagram/min_max_pd", { diracs[ i ], Task::ref_num( dim ) } );
    while( min_maxs.size() > 1 ) {
        std::vector<TaskRef> new_min_maxs( ( min_maxs.size() + 3 ) / 4 );
        for( std::size_t i = 0; i < new_min_maxs.size(); ++i ) {
            std::vector<TaskRef> inputs;
            inputs.reserve( 4 );
            for( std::size_t j = 4 * i; j < min( 4 * i + 4, min_maxs.size() ); ++j )
                inputs.push_back( min_maxs[ j ] );
            new_min_maxs[ i ] = Task::call_r( "sdot/geometry/kernels/PowerDiagram/min_max_pd_red", std::move( inputs ) );
        }
        std::swap( new_min_maxs, min_maxs );
    }

    f( std::move( min_maxs[ 0 ] ) );
}

} // namespace sdot
