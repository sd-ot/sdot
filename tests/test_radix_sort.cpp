#include <parex/containers/Vec.h>
#include <parex/support/P.h>
#include <parex/Value.h>

using namespace parex;

using TI = std::uint64_t;
using TF = double;

int main() {
    int max_value = 10;
    std::vector<TaskRef> values;
    for( std::size_t i = 0; i < 4; ++i ) {
        Vec<int> *loc = new Vec<int>( 10 );
        for( std::size_t i = 0; i < loc->size(); ++i )
            (*loc)[ i ] = rand() % max_value;
        values.push_back( Task::ref_on( loc ) );
    }

    // counts
    std::vector<TaskRef> counts;
    Value v_max_value = max_value;
    for( std::size_t i = 0; i < values.size(); ++i )
        counts.push_back( Task::call_r( "parex/kernels/count_int_lane", { values[ i ], v_max_value.ref } ) );

    // weights
    Value weights = Vec<int>{ 2, 1 };

    //
    std::vector<TaskRef> scan_args;
    scan_args.push_back( weights.ref );
    scan_args.push_back( v_max_value.ref );
    for( const TaskRef &value : values )
        scan_args.push_back( value );
    Value s = Task::call_r( Kernel::with_vararg_num( 2, {}, {}, "parex/kernels/scan_lane_weighted" ), std::move( scan_args ) );
    P( s );
}
