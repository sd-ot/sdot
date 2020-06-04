#include <boost/multiprecision/cpp_int.hpp>
#include "ZGridDiracSetStdFactory.h"
#include "internal/ZCoords.h"
#include "../support/Void.h"
#include "../support/P.h"
#include "ZGrid.h"
#define ZGrid SDOT_CONCAT_TOKEN_4( ZGrid_, DIM, _, PROFILE )

namespace sdot {

static ZGridDiracSetStdFactory<ARCH,TF,ST,DIM,Void> zdssf;

ZGrid::ZGrid( ZGridDiracSetFactory<TF,ST> *dirac_set_factory ) {
    if ( ! dirac_set_factory ) dirac_set_factory = &zdssf;
    this->dirac_set_factory = dirac_set_factory;

    available_memory = 16e9;

    // max_dirac_per_sst = dirac_set_factory->nb_diracs_for_mem( 1e9 /* 1 Go */ );
}

ZGrid::~ZGrid() {
}

void ZGrid::update( const std::function<void( const sdot::ZGrid::CbConstruct & )> &f, const UpdateParms &update_parms ) {
    using std::max;

    // min_point, max_point
    ST approx_nb_diracs = 0;
    if ( has_nan( update_parms.hist_min_point ) || has_nan( update_parms.hist_max_point ) || update_parms.approx_nb_diracs == 0 ) {
        get_min_and_max_pts( f, update_parms, approx_nb_diracs );
    } else {
        approx_nb_diracs = update_parms.approx_nb_diracs;
        min_point = update_parms.hist_min_point;
        max_point = update_parms.hist_max_point;

        grid_length = max( max_point - min_point ) * ( 1 + std::numeric_limits<TF>::epsilon() );
        step_length = grid_length / ( TZ( 1 ) << nb_bits_per_axis );
        inv_step_length = TF( 1 ) / step_length;
    }

    // histogram
    update_histogram( f, update_parms, approx_nb_diracs );

    // make the boxes
    make_boxes_rec( f, update_parms );

}

void ZGrid::get_min_and_max_pts( const std::function<void( const ZGrid::CbConstruct &)> &f, const UpdateParms &/*update_parms*/, ST &nb_diracs ) {
    using std::min;
    using std::max;

    // traversal
    min_point = + std::numeric_limits<TF>::max();
    max_point = - std::numeric_limits<TF>::max();
    f( [&]( std::array<const TF *,DIM> coords, const TF */*weights*/, const ST */*ids*/, ST nb ) {
        if ( nb == 0 )
            return;

        // TODO: same thing in parallel
        for( ST dim = 0; dim < DIM; ++dim ) {
            for( ST num_dirac = 0; num_dirac < nb; ++num_dirac ) {
                min_point[ dim ] = min( min_point[ dim ], coords[ dim ][ num_dirac ] );
                max_point[ dim ] = max( max_point[ dim ], coords[ dim ][ num_dirac ] );
            }
        }

        nb_diracs += nb;
    } );

    // grid size
    grid_length = max( max_point - min_point ) * ( 1 + std::numeric_limits<TF>::epsilon() );
    step_length = grid_length / ( TZ( 1 ) << nb_bits_per_axis );
    inv_step_length = TF( 1 ) / step_length;
}

void ZGrid::update_histogram( const std::function<void(const ZGrid::CbConstruct &)> &f, const UpdateParms &update_parms, ST approx_nb_diracs ) {
    using std::min;

    histogram.resize( 1 );

    std::size_t base_size = min( available_memory / ( 2 * sizeof( SI ) ), std::size_t( approx_nb_diracs * update_parms.hist_ratio ) );
    histogram[ 0 ].resize( pow_2_le( base_size ) );
    for( SI &v : histogram[ 0 ] )
        v = 0;

    nb_diracs = 0;

    f( [&]( std::array<const TF *,DIM> coords, const TF */*weights*/, const ST */*ids*/, ST nb_diracs ) {
        // TODO: same thing in parallel
        for( ST num_dirac = 0; num_dirac < nb_diracs; ++num_dirac ) {
            TZ z = zcoords_for<TZ,nb_bits_per_axis>( coords, num_dirac, min_point, inv_step_length );

            using namespace boost::multiprecision;
            histogram[ 0 ][ std::size_t( int128_t( z ) * base_size / max_zcoords ) ]++;
        }
    } );
}

void ZGrid::make_boxes_rec( const std::function<void( const CbConstruct & )> &f, const UpdateParms &update_parms ) {
}

} // namespace sdot
