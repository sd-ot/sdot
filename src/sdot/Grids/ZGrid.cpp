#include "ZGridDiracSetStdFactory.h"
#include "../support/Void.h"
#include "../support/P.h"
#include "ZGrid.h"
#include <limits>
#define ZGrid SDOT_CONCAT_TOKEN_4( ZGrid_, DIM, _, PROFILE )

namespace sdot {

static ZGridDiracSetStdFactory<Void> zdssf;

ZGrid::ZGrid( ZGridDiracSetFactory *dirac_set_factory ) {
    if ( ! dirac_set_factory ) dirac_set_factory = &zdssf;
    this->dirac_set_factory = dirac_set_factory;
}

ZGrid::~ZGrid() {
}

void ZGrid::update( const std::function<void( const sdot::ZGrid::CbConstruct & )> &f ) {
    get_dims( f );

    P( min_point );
    P( max_point );
}

void ZGrid::get_dims( const std::function<void( const ZGrid::CbConstruct &)> &f ) {
    using std::min;
    using std::max;

    // reset
    all_ptrs_survive_the_call = true;
    ptrs_of_previous_call.clear();
    nb_diracs = 0;

    // traversal
    min_point = + std::numeric_limits<TF>::max();
    max_point = - std::numeric_limits<TF>::max();
    f( [&]( std::array<TF *,DIM> coords, const TF *weights, const ST *ids, ST nb_diracs, bool ptrs_survive_the_call ) {
        if ( nb_diracs == 0 )
            return;

        // TODO: in parallel
        for( ST dim = 0; dim < DIM; ++dim ) {
            for( ST num_dirac = 0; num_dirac < nb_diracs; ++num_dirac ) {
                min_point[ dim ] = min( min_point[ dim ], coords[ dim ][ num_dirac ] );
                max_point[ dim ] = max( max_point[ dim ], coords[ dim ][ num_dirac ] );
            }
        }

        //
        if ( ptrs_survive_the_call )
            ptrs_of_previous_call.push_back( { coords, weights, ids, nb_diracs } );
        else
            all_ptrs_survive_the_call = false;

        this->nb_diracs += nb_diracs;
    } );

    // grid size
    grid_length = 0;
    for( std::size_t d = 0; d < dim; ++d )
        grid_length = max( grid_length, max_point[ d ] - min_point[ d ] );
    grid_length *= 1 + std::numeric_limits<TF>::epsilon();

    step_length = grid_length / ( TZ( 1 ) << nb_bits_per_axis );
    inv_step_length = TF( 1 ) / step_length;
}

} // namespace sdot
