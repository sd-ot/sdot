#include "../src/sdot/PowerDiagram/get_der_centroids_and_integrals_wrt_weight_and_positions.h"
#include "../src/sdot/Domains/ConvexPolyhedronAssembly.h"
#include "../src/sdot/PowerDiagram/Visitors/SpZGrid.h"
#include "../src/sdot/PowerDiagram/get_integrals.h"
#include "../src/sdot/PowerDiagram/get_centroids.h"
#include "../src/sdot/PowerDiagram/display_vtk.h"
#include "../src/sdot/Support/Stream.h"
// #include "catch_main.h"

#include <boost/multiprecision/mpfr.hpp>

//// nsmake cpp_flag -march=native
//// nsmake lib_name gmp
//// nsmake lib_name mpfr
using namespace sdot;

template<class Pc,class Rf>
void test( typename Pc::TF epsilon, Rf radial_func, std::vector<Point2<typename Pc::TF>> ref_positions, std::vector<typename Pc::TF> ref_weights ) {
    const std::size_t nd = Pc::dim + 1;
    using Bounds = ConvexPolyhedronAssembly<Pc>;
    using Grid = SpZGrid<Pc>;
    using TF = typename Pc::TF;
    using TI = typename Pc::TI;
    using Pt = Point2<TF>;

    Bounds bounds;
    bounds.add_box( { 0, 0 }, { 1, 1 }, 1.0, -1 );

    Grid grid;
    grid.update( ref_positions.data(), ref_weights.data(), ref_weights.size() );

    std::vector<Pt> ref_centroids( ref_weights.size() );
    std::vector<TF> ref_integrals( ref_weights.size() );
    std::vector<TF> values_ap( nd * ref_weights.size() );
    get_centroids( grid, bounds, ref_positions.data(), ref_weights.data(), ref_weights.size(), radial_func, [&]( auto c, auto m, auto n ) {
        ref_integrals[ n ] = m;
        ref_centroids[ n ] = c;

        for( std::size_t d = 0; d < Pc::dim; ++d )
            values_ap[ nd * n + d ] = c[ d ];
        values_ap[ nd * n + Pc::dim ] = m;
    } );

    // ap
    std::vector<std::vector<TF>> derivatives_ap( nd * ref_weights.size() );
    for( std::size_t n = 0; n < nd * ref_weights.size(); ++n )
        derivatives_ap[ n ].resize( nd * ref_weights.size(), 0 );

    for( std::size_t c = 0; c < nd * ref_weights.size(); ++c ) {
        std::vector<TF> new_weights = ref_weights;
        std::vector<Pt> new_positions = ref_positions;
        ( c % nd == Pc::dim ? new_weights[ c / nd ] : new_positions[ c / nd ][ c % nd ] ) += epsilon;

        std::vector<TF> new_integrals( ref_weights.size() );
        std::vector<Pt> new_centroids( ref_weights.size() );
        get_centroids( grid, bounds, new_positions.data(), new_weights.data(), new_weights.size(), radial_func, [&]( auto c, auto m, auto n ) {
            new_integrals[ n ] = m;
            new_centroids[ n ] = c;
        } );

        for( std::size_t r = 0; r < nd * ref_weights.size(); ++r ) {
            derivatives_ap[ r ][ c ] = r % nd == Pc::dim ?
                  ( new_integrals[ r / nd ]           - ref_integrals[ r / nd ]           ) / epsilon :
                  ( new_centroids[ r / nd ][ r % nd ] - ref_centroids[ r / nd ][ r % nd ] ) / epsilon;
        }

    }

    for( std::size_t r = 0; r < nd * ref_weights.size(); ++r )
        P( derivatives_ap[ r ] );
    P( values_ap );

    std::vector<TI> m_offsets;
    std::vector<TI> m_columns;
    std::vector<TF> m_values;
    std::vector<TF> v_values;
    get_der_centroids_and_integrals_wrt_weight_and_positions( m_offsets, m_columns, m_values, v_values, grid, bounds, ref_positions.data(), ref_weights.data(), ref_weights.size(), radial_func );

    // ex
    std::vector<std::vector<TF>> derivatives_ex( nd * ref_weights.size() );
    for( std::size_t n = 0; n < nd * ref_weights.size(); ++n )
        derivatives_ex[ n ].resize( nd * ref_weights.size(), 0 );
    for( std::size_t r = 0; r + 1 < m_offsets.size(); ++r )
        for( std::size_t k = m_offsets[ r ]; k < m_offsets[ r + 1 ]; ++k )
            derivatives_ex[ r ][ m_columns[ k ] ] = m_values[ k ];

    for( std::size_t r = 0; r < nd * ref_weights.size(); ++r )
        P( derivatives_ex[ r ] );
    P( v_values );

    TF err = 0;
    for( std::size_t r = 0; r < nd * ref_weights.size(); ++r )
        for( std::size_t c = 0; c < nd * ref_weights.size(); ++c )
            err = max( err, abs( derivatives_ex[ r ][ c ] - derivatives_ap[ r ][ c ] ) );
    for( std::size_t r = 0; r < nd * ref_weights.size(); ++r )
        err = max( err, abs( v_values[ r ] - values_ap[ r ] ) );
    P( err );

    VtkOutput<1,TF> vo( { "num" } );
    display_vtk( vo, grid, bounds, ref_positions.data(), ref_weights.data(), ref_weights.size(), radial_func );
    vo.save( "lc.vtk" );
}

int main() {
    struct Pc { enum { dim = 2, allow_ball_cut = 1, allow_translations = 0 }; using TI = std::size_t; using TF = boost::multiprecision::mpfr_float_100; };
    using TF = typename Pc::TF;
    using Pt = Point2<TF>;
    TF epsilon = 1e-50;


    //    std::vector<Pt> ref_positions{ Pt{ 0.25, 0.1 }, Pt{ 0.75, 0.9 } };
    //    std::vector<TF> ref_weights{ 0.225, 0.32 };
    //        std::vector<Pt> ref_positions{ Pt{ 0.5, 0.5 } };
    //        std::vector<TF> ref_weights{ 0.2 };
    //    std::vector<Pt> ref_positions;
    //    std::vector<TF> ref_weights;
    //    for( std::size_t i = 0; i < 20; ++i ) {
    //        ref_positions.push_back( Pt{ TF( 1.0 * rand() / RAND_MAX ), TF( 1.0 * rand() / RAND_MAX ) } );
    //        ref_weights.push_back( TF( 0.005 + 0.005 * rand() / RAND_MAX ) );
    //    }

    //    test<Pc>( epsilon, FunctionEnum::InBallW05(), { Pt{ 0.125, 0.125}, Pt{ 0.375, 0.125}, Pt{ 0.125, 0.375}, Pt{ 0.375, 0.375} }, { 0.02490201, 0.02289457, 0.02289457, 0.02162449 } );

    std::vector<Pt> ref_positions{
        Pt{ 0.05, 0.05 }, Pt{ 0.13, 0.05 }, Pt{ 0.21, 0.05 }, Pt{ 0.29, 0.05 }, Pt{ 0.37, 0.05 },
        Pt{ 0.45, 0.05 }, Pt{ 0.05, 0.13 }, Pt{ 0.13, 0.13 }, Pt{ 0.21, 0.13 }, Pt{ 0.29, 0.13 },
        Pt{ 0.37, 0.13 }, Pt{ 0.45, 0.13 }, Pt{ 0.05, 0.21 }, Pt{ 0.13, 0.21 }, Pt{ 0.21, 0.21 },
        Pt{ 0.29, 0.21 }, Pt{ 0.37, 0.21 }, Pt{ 0.45, 0.21 }, Pt{ 0.05, 0.29 }, Pt{ 0.13, 0.29 },
        Pt{ 0.21, 0.29 }, Pt{ 0.29, 0.29 }, Pt{ 0.37, 0.29 }, Pt{ 0.45, 0.29 }, Pt{ 0.05, 0.37 },
        Pt{ 0.13, 0.37 }, Pt{ 0.21, 0.37 }, Pt{ 0.29, 0.37 }, Pt{ 0.37, 0.37 }, Pt{ 0.45, 0.37 },
        Pt{ 0.05, 0.45 }, Pt{ 0.13, 0.45 }, Pt{ 0.21, 0.45 }, Pt{ 0.29, 0.45 }, Pt{ 0.37, 0.45 },
        Pt{ 0.45, 0.45 }
    };
    std::vector<TF> ref_weights{
        0.00314773, 0.00393608, 0.00428933, 0.0042071 , 0.00371394, 0.00284918,
        0.00393608, 0.0048636 , 0.00525743, 0.00514763, 0.00452475, 0.00332225,
        0.00428933, 0.00525743, 0.00566234, 0.0055339 , 0.00484392, 0.0034862 ,
        0.0042071 , 0.00514763, 0.0055339 , 0.00541476, 0.00476184, 0.00345417,
        0.00371394, 0.00452475, 0.00484392, 0.00476184, 0.00426395, 0.00322731,
        0.00284918, 0.00332225, 0.0034862 , 0.00345417, 0.00322731, 0.00270565
    };
    test<Pc>( epsilon, FunctionEnum::InBallW05(), ref_positions, ref_weights );
}
