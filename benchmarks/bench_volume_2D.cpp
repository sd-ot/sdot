#include "../src/sdot/bounds/ConvexPolyhedronAssembly.h"
#include "../src/sdot/visitors/SpZGrid.h"
#include "../src/sdot/visitors/ZGrid.h"
#include "../src/sdot/system/MpiInst.h"
#include "../src/sdot/system/Time.h"
#include "set_up_diracs_2D.h"
#include <cxxopts.hpp>

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -ffast-math
//// nsmake cpp_flag -O5
//// nsmake lib_flag -O5

int main( int argc, char **argv ) {
    struct Pc { enum { nb_bits_per_axis = 31, allow_ball_cut = 0, dim = 2 }; using TI = std::size_t; using TF = double; };
    using  Pt = Point2<Pc::TF>;
    using  TF = Pc::TF;

    // options
    cxxopts::Options options( argv[ 0 ], "bench volume");
    options.add_options()
        ( "m,max-dirac-per-cell"    , "...", cxxopts::value<int>()->default_value( "11" ) )
        ( "r,max-delta-weight"      , "...", cxxopts::value<double>()->default_value( "1e40" ) )
        ( "eq-w-repartition"        , "..." )
        ( "d,distribution"          , "distribution name (regular, random, ...)", cxxopts::value<std::string>()->default_value( "regular" ) )
        ( "t,nb-threads"            , "...", cxxopts::value<int>()->default_value( "0" ) )
        ( "v,vtk-output"            , "", cxxopts::value<std::string>() )
        ( "n,nb-diracs"             , "...", cxxopts::value<double>()->default_value( "100" ) )
        ( "o,output"                , "", cxxopts::value<std::string>() )
        ( "periodic"                , "" )
        ( "h,help"                  , "get help" )
        ;
    auto args = options.parse( argc, argv );

    //
    thread_pool.init( args[ "nb-threads" ].as<int>() );
    mpi_inst.init( argc, argv );

    // diracs
    std::vector<TF> weights;
    std::vector<Pt> positions;
    set_up_diracs( positions, weights, args[ "distribution" ].as<std::string>(), args[ "nb-diracs" ].as<double>() );

    // grid
    #ifdef USE_ZGRID
    using Grid = sdot::ZGrid<Pc>;
    Grid grid( args[ "max-dirac-per-cell" ].as<int>(), args[ "max-delta-weight" ].as<double>() );
    grid.eq_rep_weight_split = args.count( "eq-w-repartition" );
    #else
    using Grid = sdot::SpZGrid<Pc>;
    Grid grid( args[ "max-dirac-per-cell" ].as<int>() );
    #endif // USE_ZGRID

    bool periodic = args.count( "periodic" );
    if ( periodic )
        for( Pc::TF y = -1; y <= 1; ++y )
            for( Pc::TF x = -1; x <= 1; ++x )
                if ( x || y )
                    grid.translations.push_back( { x, y } );

    // Bounds
    using Bounds = sdot::ConvexPolyhedronAssembly<Pc>;
    Bounds bounds;
    bounds.add_box( { - TF( periodic ), - TF( periodic ) }, { 1 + TF( periodic ), 1 + TF( periodic ) } );

    // volume
    std::vector<TF> volumes( weights.size(), TF( 0 ) );
    auto t0 = Time::get_time();
    grid.update( positions.data(), weights.data(), weights.size() );
    auto t1 = Time::get_time();
    grid.for_each_laguerre_cell( [&]( auto &lc, std::size_t num_dirac ) {
        volumes[ num_dirac ] = lc.measure();
    }, bounds.englobing_convex_polyhedron(), positions.data(), weights.data(), weights.size() );
    auto t2 = Time::get_time();

    TF vol;
    for( TF v : volumes )
        vol += v;
    PMPI( weights.size() );
    PMPI_0( mpi->reduction( vol, [](double a, double b ) { return a + b; } ) );
    PMPI_0( Time::delta( t0, t1 ) );
    PMPI_0( Time::delta( t1, t2 ) );
    PMPI_0( Time::delta( t0, t2 ) );

    // display
    if ( args.count( "vtk-output" ) ) {
        VtkOutput<2> vtk_output( { "weight", "num" } );

        grid.update( positions.data(), weights.data(), weights.size() );
        grid.for_each_laguerre_cell( [&]( auto &lc, std::size_t num_dirac ) {
            lc.display( vtk_output, { weights[ num_dirac ], TF( num_dirac ) } );
        }, bounds.englobing_convex_polyhedron(), positions.data(), weights.data(), weights.size() );

        vtk_output.save( args[ "vtk-output" ].as<std::string>() + "_" + to_string( mpi->rank() ) + ".vtk" );
    }
}
