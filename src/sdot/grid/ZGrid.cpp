//#include <boost/multiprecision/cpp_int.hpp>
//#include "ZGridDiracSetStdFactory.h"
//#include "internal/ZCoords.h"
//#include "../support/Void.h"
//#include "../support/TODO.h"
//#include "../support/P.h"
//#include "ZGrid.h"
//#define ZGrid SDOT_CONCAT_TOKEN_4( ZGrid_, DIM, _, PROFILE )

//namespace sdot {

//static ZGridDiracSetStdFactory<ARCH,TF,ST,DIM,Void> zdssf;

//ZGrid::ZGrid( ZGridDiracSetFactory<TF,ST> *dirac_set_factory ) {
//    if ( ! dirac_set_factory ) dirac_set_factory = &zdssf;
//    this->dirac_set_factory = dirac_set_factory;

//    max_nb_diracs_per_cell = 30;
//    available_memory = 16e9;
//}

//ZGrid::~ZGrid() {
//}

//void ZGrid::write_to_stream( std::ostream &os ) const {
//    os << "boxes:\n";
//    for( ST nb = 0; nb < nb_boxes; ++nb )
//        write_box_to_stream( os, boxes[ nb ], "  " );
//}

//void ZGrid::update( const std::function<void( const sdot::ZGrid::CbConstruct & )> &f, const UpdateParms &update_parms ) {
//    using std::max;

//    // min_point, max_point
//    ST approx_nb_diracs = 0;
//    if ( has_nan( update_parms.hist_min_point ) || has_nan( update_parms.hist_max_point ) || update_parms.approx_nb_diracs == 0 ) {
//        get_min_and_max_pts( f, update_parms, approx_nb_diracs );
//    } else {
//        approx_nb_diracs = update_parms.approx_nb_diracs;
//        min_point = update_parms.hist_min_point;
//        max_point = update_parms.hist_max_point;

//        grid_length = max( max_point - min_point ) * ( 1 + std::numeric_limits<TF>::epsilon() );
//        step_length = grid_length / ( TZ( 1 ) << nb_bits_per_axis );
//        inv_step_length = TF( 1 ) / step_length;
//    }

//    // histogram
//    update_histogram( f, update_parms, approx_nb_diracs );

//    // make the boxes (using the histogram)
//    make_the_boxes( f, update_parms );

//    //
//    fill_the_boxes( f, update_parms );
//}

//void ZGrid::get_min_and_max_pts( const std::function<void( const ZGrid::CbConstruct &)> &f, const UpdateParms &/*update_parms*/, ST &nb_diracs ) {
//    using std::min;
//    using std::max;

//    // traversal
//    min_point = + std::numeric_limits<TF>::max();
//    max_point = - std::numeric_limits<TF>::max();
//    f( [&]( std::array<const TF *,DIM> coords, const TF */*weights*/, const ST */*ids*/, ST nb ) {
//        if ( nb == 0 )
//            return;

//        // TODO: same thing in parallel
//        for( ST dim = 0; dim < DIM; ++dim ) {
//            for( ST num_dirac = 0; num_dirac < nb; ++num_dirac ) {
//                min_point[ dim ] = min( min_point[ dim ], coords[ dim ][ num_dirac ] );
//                max_point[ dim ] = max( max_point[ dim ], coords[ dim ][ num_dirac ] );
//            }
//        }

//        nb_diracs += nb;
//    } );

//    // grid size
//    grid_length = max( max_point - min_point ) * ( 1 + std::numeric_limits<TF>::epsilon() );
//    step_length = grid_length / ( TZ( 1 ) << nb_bits_per_axis );
//    inv_step_length = TF( 1 ) / step_length;
//}

//void ZGrid::update_histogram( const std::function<void(const ZGrid::CbConstruct &)> &f, const UpdateParms &update_parms, ST approx_nb_diracs ) {
//    using std::min;

//    histograms.resize( 1 );

//    std::size_t base_size = pow_2_le( min( available_memory / ( 2 * sizeof( SI ) ), std::size_t( approx_nb_diracs * update_parms.hist_ratio ) ) );
//    histograms[ 0 ].resize( base_size + 1 );
//    for( SI &v : histograms[ 0 ] )
//        v = 0;

//    nb_diracs = 0;

//    f( [&]( std::array<const TF *,DIM> coords, const TF */*weights*/, const ST */*ids*/, ST nb_diracs ) {
//        // TODO: same thing in parallel
//        for( ST num_dirac = 0; num_dirac < nb_diracs; ++num_dirac ) {
//            TZ z = zcoords_for<TZ,nb_bits_per_axis>( coords, num_dirac, min_point, inv_step_length );

//            using namespace boost::multiprecision;
//            histograms[ 0 ][ std::size_t( int128_t( z ) * base_size / max_zcoords ) ]++;
//        }
//    } );
//}

//void ZGrid::make_the_boxes( const std::function<void( const CbConstruct & )> &f, const UpdateParms &update_parms ) {
//    // accumulation
//    std::vector<SI> &h = histograms[ 0 ];
//    ST base_size = h.size() - 1;
//    for( ST acc = 0, ind = 0; ind < h.size(); ++ind ) {
//        SI val = h[ ind ];
//        h[ ind ] = acc;
//        acc += val;
//    }

//    final_boxes.resize( histograms.size() );
//    for( ST i = 0; i < histograms.size(); ++i )
//        final_boxes[ i ].resize( histograms[ i ].size() - 1 );

//    //
//    nb_boxes = 0;

//    box_pool.clear();

//    ST beg_h = 0;
//    ST end_h = h.size() - 1;
//    ST m_0_h = ST( beg_h + ( end_h - beg_h ) * 1 / 4 );
//    ST m_1_h = ST( beg_h + ( end_h - beg_h ) * 2 / 4 );
//    ST m_2_h = ST( beg_h + ( end_h - beg_h ) * 3 / 4 );
//    make_the_boxes_rec( boxes, nb_boxes, h, beg_h, m_0_h, 0, max_zcoords / base_size );
//    make_the_boxes_rec( boxes, nb_boxes, h, m_0_h, m_1_h, 0, max_zcoords / base_size );
//    make_the_boxes_rec( boxes, nb_boxes, h, m_1_h, m_2_h, 0, max_zcoords / base_size );
//    make_the_boxes_rec( boxes, nb_boxes, h, m_2_h, end_h, 0, max_zcoords / base_size );
//}

//void ZGrid::make_the_boxes_rec( Box **boxes, ST &nb_boxes, const std::vector<SI> &h, ST beg_h, ST end_h, ST off_h, ST mul_h ) {
//    if ( end_h - beg_h == 0 )
//        return;

//    ST nb_diracs_loc = h[ end_h ] - h[ beg_h ];
//    if ( nb_diracs_loc == 0 )
//        return;

//    // we create a new box in any cases
//    Box *box = box_pool.create<Box>();
//    boxes[ nb_boxes++ ] = box;

//    box->beg_zcoord = off_h + beg_h * mul_h;
//    box->end_zcoord = off_h + end_h * mul_h;

//    box->min_point = + std::numeric_limits<TF>::max();
//    box->max_point = - std::numeric_limits<TF>::max();

//    if ( w_bounds_order == 0 ) {
//        box->w_bound[ 0 ] = - std::numeric_limits<TF>::max();
//    } else {
//        TODO;
//    }

//    // final box ?
//    if ( h[ end_h ] - h[ beg_h ] <= SI( max_nb_diracs_per_cell ) || end_h - beg_h == 1 ) {
//        box->dirac_set = dirac_set_factory->New( box_pool, nb_diracs_loc );
//        for( ST ind = beg_h; ind < end_h; ++ind )
//            final_boxes[ 0 ][ ind ] = box;
//        return;
//    }

//    // else, make sub boxes
//    ST m_0_h = ST( beg_h + std::uint64_t( end_h - beg_h ) * 1 / 4 );
//    ST m_1_h = ST( beg_h + std::uint64_t( end_h - beg_h ) * 2 / 4 );
//    ST m_2_h = ST( beg_h + std::uint64_t( end_h - beg_h ) * 3 / 4 );
//    make_the_boxes_rec( box->sub_boxes, box->nb_sub_boxes, h, beg_h, m_0_h, off_h, mul_h );
//    make_the_boxes_rec( box->sub_boxes, box->nb_sub_boxes, h, m_0_h, m_1_h, off_h, mul_h );
//    make_the_boxes_rec( box->sub_boxes, box->nb_sub_boxes, h, m_1_h, m_2_h, off_h, mul_h );
//    make_the_boxes_rec( box->sub_boxes, box->nb_sub_boxes, h, m_2_h, end_h, off_h, mul_h );
//}

//void ZGrid::fill_the_boxes( const std::function<void( const CbConstruct & )> &f, const UpdateParms &update_parms ) {
//    using std::max;

//    ST base_size = histograms[ 0 ].size() - 1;

//    // push diracs and update min and max points in final boxes
//    f( [&]( std::array<const TF *,DIM> coords, const TF *weights, const ST *ids, ST nb_diracs ) {
//        // TODO: same thing in parallel
//        for( ST num_dirac = 0; num_dirac < nb_diracs; ++num_dirac ) {
//            TZ z = zcoords_for<TZ,nb_bits_per_axis>( coords, num_dirac, min_point, inv_step_length );

//            Pt c;
//            for( ST d = 0; d < dim; ++d )
//                c[ d ] = coords[ d ][ num_dirac ];

//            Box *fb = final_boxes[ 0 ][ z / ( max_zcoords / base_size ) ];
//            fb->dirac_set->add_dirac( c.data, weights[ num_dirac ], ids[ num_dirac ] );
//            fb->min_point = min( fb->min_point, c );
//            fb->max_point = max( fb->max_point, c );

//            if ( w_bounds_order == 0 ) {
//                fb->w_bound[ 0 ] = max( fb->w_bound[ 0 ], weights[ num_dirac ] );
//            } else {
//                TODO;
//            }
//        }
//    } );

//    // update min and max points in non final boxes
//    for_each_box( [&]( Box *box ) {
//        if ( box->dirac_set )
//            return;
//        box->min_point = box->sub_boxes[ 0 ]->min_point;
//        box->max_point = box->sub_boxes[ 0 ]->max_point;
//        if ( w_bounds_order == 0 ) {
//            box->w_bound[ 0 ] = box->sub_boxes[ 0 ]->w_bound[ 0 ];
//        } else {
//            TODO;
//        }
//        for( ST i = 1; i < box->nb_sub_boxes; ++i ) {
//            box->min_point = min( box->min_point, box->sub_boxes[ i ]->min_point );
//            box->max_point = max( box->max_point, box->sub_boxes[ i ]->max_point );
//            if ( w_bounds_order == 0 ) {
//                box->w_bound[ 0 ] = max( box->w_bound[ 0 ], box->sub_boxes[ i ]->w_bound[ 0 ] );
//            } else {
//                TODO;
//            }
//        }
//    }, boxes, nb_boxes );
//}

//void ZGrid::for_each_box( const std::function<void( Box *)> &f, Box **boxes, ST nb_boxes ) {
//    for( ST i = 0; i < nb_boxes; ++i ) {
//        for_each_box( f, boxes[ i ]->sub_boxes, boxes[ i ]->nb_sub_boxes );
//        f( boxes[ i ] );
//    }
//}

//void ZGrid::write_box_to_stream( std::ostream &os, const Box *box, std::string sp ) const {
//    os << sp << "z: " << box->beg_zcoord << " -> " << box->end_zcoord << "; p: " << box->min_point << ", " << box->max_point << "\n";
//    if ( box->dirac_set )
//        box->dirac_set->write_to_stream( os, sp + "  " );
//    for( ST i = 0; i < box->nb_sub_boxes; ++i )
//        write_box_to_stream( os, box->sub_boxes[ i ], sp + "  " );
//}

//} // namespace sdot
