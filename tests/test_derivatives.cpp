//#include "../src/sdot/PowerDiagram/get_der_boundary_measure.h"
#include "../src/sdot/PowerDiagram/get_der_boundary_integral.h"
#include "../src/sdot/Domains/ConvexPolyhedronAssembly.h"
#include "../src/sdot/PowerDiagram/Visitors/SpZGrid.h"
#include "../src/sdot/PowerDiagram/display_vtk.h"
#include "../src/sdot/Support/Stream.h"
// #include "catch_main.h"

// #include <boost/multiprecision/mpfr.hpp>

//// nsmake cpp_flag -march=native
// //// nsmake lib_name gmp
// //// nsmake lib_name mpfr
using namespace sdot;

int main() {
    struct Pc { enum { dim = 2, allow_ball_cut = 1, allow_translations = 0 }; using TI = std::size_t; using TF = double/*boost::multiprecision::mpfr_float_100*/; };
    using Bounds = ConvexPolyhedronAssembly<Pc>;
    using Grid = SpZGrid<Pc>;
    using std::pow;
    using std::cos;
    using std::sin;

    using Pt = typename Grid::Pt;
    using CP = typename Grid::CP;
    using TF = typename Pc::TF;
    using TI = typename Pc::TI;

    FunctionEnum::InBallW05 radial_func;

    for( std::size_t n : { 20 } ) {
        Bounds bounds;
        bounds.add_box( { -10, -10 }, { 10, 10 } );

        std::vector<Pt> positions{ Pt{ 0, 0 } };
        std::vector<TF> weights{ 1.0 };
        for( std::size_t i = 0; i < n; ++i ) {
            TF d = 2 * CP::pi() / n, a = d * i;
            positions.push_back( Pt{ cos( a ), sin( a ) } );
            weights.push_back( pow( d / 2, 2 ) );
        }

        Grid grid;
        grid.update( positions.data(), weights.data(), weights.size() );
        P( get_boundary_integral( grid, bounds, positions.data(), weights.data(), weights.size(), radial_func ) );

        VtkOutput<1,TF> vo( { "num" } );
        display_vtk( vo, grid, bounds, positions.data(), weights.data(), weights.size(), radial_func );
        vo.save( "lc.vtk" );

    }

    //    std::vector<Pt> ref_centroids( ref_weights.size() );
    //    std::vector<TF> ref_integrals( ref_weights.size() );
    //    std::vector<TF> values_ap( nd * ref_weights.size() );
    //    get_centroids( grid, bounds, ref_positions.data(), ref_weights.data(), ref_weights.size(), radial_func, [&]( auto c, auto m, auto n ) {
    //        ref_integrals[ n ] = m;
    //        ref_centroids[ n ] = c;

    //        for( std::size_t d = 0; d < Pc::dim; ++d )
    //            values_ap[ nd * n + d ] = c[ d ];
    //        values_ap[ nd * n + Pc::dim ] = m;
    //    } );

    //    // ap
    //    std::vector<std::vector<TF>> derivatives_ap( nd * ref_weights.size() );
    //    for( std::size_t n = 0; n < nd * ref_weights.size(); ++n )
    //        derivatives_ap[ n ].resize( nd * ref_weights.size(), 0 );

    //    for( std::size_t c = 0; c < nd * ref_weights.size(); ++c ) {
    //        std::vector<TF> new_weights = ref_weights;
    //        std::vector<Pt> new_positions = ref_positions;
    //        ( c % nd == Pc::dim ? new_weights[ c / nd ] : new_positions[ c / nd ][ c % nd ] ) += epsilon;

    //        std::vector<TF> new_integrals( ref_weights.size() );
    //        std::vector<Pt> new_centroids( ref_weights.size() );
    //        get_centroids( grid, bounds, new_positions.data(), new_weights.data(), new_weights.size(), radial_func, [&]( auto c, auto m, auto n ) {
    //            new_integrals[ n ] = m;
    //            new_centroids[ n ] = c;
    //        } );

    //        for( std::size_t r = 0; r < nd * ref_weights.size(); ++r ) {
    //            derivatives_ap[ r ][ c ] = r % nd == Pc::dim ?
    //                  ( new_integrals[ r / nd ]           - ref_integrals[ r / nd ]           ) / epsilon :
    //                  ( new_centroids[ r / nd ][ r % nd ] - ref_centroids[ r / nd ][ r % nd ] ) / epsilon;
    //        }

    //    }

    //    for( std::size_t r = 0; r < nd * ref_weights.size(); ++r )
    //        P( derivatives_ap[ r ] );
    //    P( values_ap );

    //    std::vector<TI> m_offsets;
    //    std::vector<TI> m_columns;
    //    std::vector<TF> m_values;
    //    std::vector<TF> v_values;
    //    get_der_centroids_and_integrals_wrt_weight_and_positions( m_offsets, m_columns, m_values, v_values, grid, bounds, ref_positions.data(), ref_weights.data(), ref_weights.size(), radial_func );

    //    // ex
    //    std::vector<std::vector<TF>> derivatives_ex( nd * ref_weights.size() );
    //    for( std::size_t n = 0; n < nd * ref_weights.size(); ++n )
    //        derivatives_ex[ n ].resize( nd * ref_weights.size(), 0 );
    //    for( std::size_t r = 0; r + 1 < m_offsets.size(); ++r )
    //        for( std::size_t k = m_offsets[ r ]; k < m_offsets[ r + 1 ]; ++k )
    //            derivatives_ex[ r ][ m_columns[ k ] ] = m_values[ k ];

    //    for( std::size_t r = 0; r < nd * ref_weights.size(); ++r )
    //        P( derivatives_ex[ r ] );
    //    P( v_values );

    //    TF err = 0;
    //    for( std::size_t r = 0; r < nd * ref_weights.size(); ++r )
    //        for( std::size_t c = 0; c < nd * ref_weights.size(); ++c )
    //            err = max( err, abs( derivatives_ex[ r ][ c ] - derivatives_ap[ r ][ c ] ) );
    //    for( std::size_t r = 0; r < nd * ref_weights.size(); ++r )
    //        err = max( err, abs( v_values[ r ] - values_ap[ r ] ) );
    //    P( err );
}
