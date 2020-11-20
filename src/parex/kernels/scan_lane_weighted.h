#include <parex/arch/SimdVecRec.h>
#include <parex/arch/SimdRange.h>
#include <parex/containers/Vec.h>
#include <parex/support/P.h>
using namespace parex;

template<class VW,class TI,class VC>
int *scan_lane_weighted( const VW &weights, TI max_value, Vec<VC *> &counts ) {
    TI nb_items = 0;
    for( VC *vc : counts )
        for( auto v : vc )
            nb_items += v;
    P( nb_items );

    //
    for( std::size_t num_machine = 0; num_machine < weights.size(); ++num_machine ) {

    }


    //
    int *res = new int( 10 );
    P( weights, max_value, counts.size() );
    return res;
}
