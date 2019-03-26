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

template<class Pc>
void test( typename Pc::TF epsilon ) {
    using Bounds = ConvexPolyhedronAssembly<Pc>;
    using Grid = SpZGrid<Pc>;
    using TF = typename Pc::TF;
    using TI = typename Pc::TI;
    using Pt = Point2<TF>;

    const std::size_t nd = Pc::dim + 1;

    //    std::vector<Pt> ref_positions{ Pt{ 0.25, 0.1 }, Pt{ 0.75, 0.9 } };
    //    std::vector<TF> ref_weights{ 1.0, 1.1 };
    std::vector<Pt> ref_positions;
    std::vector<TF> ref_weights;
    for( std::size_t i = 0; i < 10; ++i ) {
        ref_positions.push_back( Pt{ TF( 1.0 * rand() / RAND_MAX ), TF( 1.0 * rand() / RAND_MAX ) } );
        ref_weights.push_back( TF( 1.0 + 0.1 * rand() / RAND_MAX ) );
    }

    Bounds bounds;
    bounds.add_box( { 0, 0 }, { 1, 1 }, 1.0, -1 );

    Grid grid;
    grid.update( ref_positions.data(), ref_weights.data(), ref_weights.size() );

    std::vector<Pt> ref_centroids( ref_weights.size() );
    std::vector<TF> ref_integrals( ref_weights.size() );
    std::vector<TF> values_ap( nd * ref_weights.size() );
    get_centroids( grid, bounds, ref_positions.data(), ref_weights.data(), ref_weights.size(), [&]( auto c, auto m, auto n ) {
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
        get_centroids( grid, bounds, new_positions.data(), new_weights.data(), new_weights.size(), [&]( auto c, auto m, auto n ) {
            new_integrals[ n ] = m;
            new_centroids[ n ] = c;
        } );

        for( std::size_t r = 0; r < nd * ref_weights.size(); ++r ) {
            derivatives_ap[ r ][ c ] = r % nd == Pc::dim ?
                  ( new_integrals[ r / nd ]           - ref_integrals[ r / nd ]           ) / epsilon :
                  ( new_centroids[ r / nd ][ r % nd ] - ref_centroids[ r / nd ][ r % nd ] ) / epsilon;
        }

    }

    //    for( std::size_t r = 0; r < nd * ref_weights.size(); ++r )
    //        P( derivatives_ap[ r ] );
    //    P( values_ap );

    std::vector<TI> m_offsets;
    std::vector<TI> m_columns;
    std::vector<TF> m_values;
    std::vector<TF> v_values;
    get_der_centroids_and_integrals_wrt_weight_and_positions( m_offsets, m_columns, m_values, v_values, grid, bounds, ref_positions.data(), ref_weights.data(), ref_weights.size() );

    // ex
    std::vector<std::vector<TF>> derivatives_ex( nd * ref_weights.size() );
    for( std::size_t n = 0; n < nd * ref_weights.size(); ++n )
        derivatives_ex[ n ].resize( nd * ref_weights.size(), 0 );
    for( std::size_t r = 0; r + 1 < m_offsets.size(); ++r )
        for( std::size_t k = m_offsets[ r ]; k < m_offsets[ r + 1 ]; ++k )
            derivatives_ex[ r ][ m_columns[ k ] ] = m_values[ k ];

    //    for( std::size_t r = 0; r < nd * ref_weights.size(); ++r )
    //        P( derivatives_ex[ r ] );
    //    P( v_values );

    TF err = 0;
    for( std::size_t r = 0; r < nd * ref_weights.size(); ++r )
        for( std::size_t c = 0; c < nd * ref_weights.size(); ++c )
            err = max( err, abs( derivatives_ex[ r ][ c ] - derivatives_ap[ r ][ c ] ) );
    for( std::size_t r = 0; r < nd * ref_weights.size(); ++r )
        err = max( err, abs( v_values[ r ] - values_ap[ r ] ) );
    P( err );

    //    VtkOutput<1,TF> vo( { "num" } );
    //    display_vtk( vo, grid, bounds, positions.data(), weights.data(), weights.size() );
    //    vo.save( "lc.vtk" );
}

int main() {
    struct Pc { enum { dim = 2, allow_ball_cut = 0, allow_translations = 0 }; using TI = std::size_t; using TF = boost::multiprecision::mpfr_float_100; };
    test<Pc>( 1e-50 );
}
