#include "ZGridDiracSetStdFactory.h"
#include "internal/ZCoords.h"
#include "../support/Void.h"
#include "../support/P.h"
#include "ZGrid.h"
#define ZGrid SDOT_CONCAT_TOKEN_4( ZGrid_, DIM, _, PROFILE )

namespace sdot {

//template<class Arch,class T,class S,int dim,class ContentByDirac>
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
    get_dimensions( f, update_parms );
    make_the_cells( f, update_parms );
}

void ZGrid::get_dimensions( const std::function<void( const ZGrid::CbConstruct &)> &f, const UpdateParms &update_parms ) {
    using std::min;
    using std::max;

    // we need

    // Le but premier est de répartir les diracs dans les machines, les out_of_core et commencer à mettre dans des sous-structures.
    nb_diracs = 0;
    if ( has_nan( update_parms.hist_min_point ) || has_nan( update_parms.hist_max_point ) ) {

    }

    // reset
    all_ptrs_survive_the_call = true;
    ptrs_of_previous_call.clear();
    hist_inv_step_length = 0;
    hist_is_done = false;
    hist.clear();

    // if user has provided incl_min_point and incl_max_point
    bool can_make_hist = true;
    for( TF v : update_parms.hist_min_point )
        can_make_hist &= v != - std::numeric_limits<TF>::max();
    for( TF v : update_parms.hist_max_point )
        can_make_hist &= v != + std::numeric_limits<TF>::max();
    if ( can_make_hist ) {
        TF hist_grid_length = 0;
        for( std::size_t d = 0; d < dim; ++d )
            hist_grid_length = max( hist_grid_length, update_parms.hist_max_point[ d ] - update_parms.hist_min_point[ d ] );
        hist_inv_step_length = ( TZ( 1 ) << nb_bits_per_axis ) / ( hist_grid_length * ( 1 + std::numeric_limits<TF>::epsilon() ) );
        hist_min_point = update_parms.hist_min_point;
        hist_max_point = update_parms.hist_max_point;
    }

    // traversal
    min_point = + std::numeric_limits<TF>::max();
    max_point = - std::numeric_limits<TF>::max();
    ST approx_nb_diracs = update_parms.approx_nb_diracs;
    f( [&]( std::array<const TF *,DIM> coords, const TF *weights, const ST *ids, ST nb_diracs, bool ptrs_survive_the_call ) {
        if ( nb_diracs == 0 )
            return;

        // TODO: same thing in parallel
        for( ST dim = 0; dim < DIM; ++dim ) {
            for( ST num_dirac = 0; num_dirac < nb_diracs; ++num_dirac ) {
                min_point[ dim ] = min( min_point[ dim ], coords[ dim ][ num_dirac ] );
                max_point[ dim ] = max( max_point[ dim ], coords[ dim ][ num_dirac ] );
            }
        }

        // storage of ptrs if relevant
        if ( ptrs_survive_the_call )
            ptrs_of_previous_call.push_back( { coords, weights, ids, nb_diracs } );
        else
            all_ptrs_survive_the_call = false;

        // histogram
        if ( can_make_hist ) {
            if ( approx_nb_diracs == 0 )
                approx_nb_diracs = nb_diracs;
            if ( hist.empty() )
                hist.resize( approx_nb_diracs / 16 + 1, 0 );

            // TODO: same thing in parallel
            for( ST num_dirac = 0; num_dirac < nb_diracs; ++num_dirac ) {
                TZ zcoords = zcoords_for_bounded<TZ,nb_bits_per_axis>( coords, num_dirac, hist_min_point, hist_max_point, hist_inv_step_length );
                ++hist[ TF( zcoords ) * hist.size() / max_zcoords ];
            }
        }

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

void ZGrid::make_the_cells( const std::function<void(const ZGrid::CbConstruct &)> &f, const UpdateParms &update_parms ) {
    // qi
    if (  ) {

    }

    // construction des cellules
}

} // namespace sdot
