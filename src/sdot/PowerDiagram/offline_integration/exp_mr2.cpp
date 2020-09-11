#include "lib/ApproximateNdIntegratorRadialFunc.h"
#include "../system/Assert.h"

template<class Func>
int make_and_test( std::string name, int degp, const Func &func ) {
    using TF = ApproximateNdIntegratorRadialFunc::TF;

    ApproximateNdIntegratorRadialFunc p( 1e-5, degp );
    p.run( func, 10, 1e3, 1e5, 1e2 );

    P( name, degp );
    //    P( p.integrate( { { TF( 0 ), TF( 0 ) }, { TF( 1 ), TF( 0 ) }, { TF( 1 ), TF( 1 ) }, { TF( 0 ), TF( 1 ) } } ) ); // 2/3
    //    P( p.integrate( { { TF( 0 ), TF( 0 ) }, { TF( 2 ), TF( 0 ) }, { TF( 2 ), TF( 1 ) }, { TF( 0 ), TF( 1 ) } } ) ); // 2/3 + 8/3

    p.generate( std::cout );
    return 0;
}

int main( int argc, char **argv ) {
    ASSERT( argc > 2, "usage: %s FunctionName polynomial_degree", argv[ 0 ] );
    std::string name = argv[ 1 ];
    int degp = atoi( argv[ 2 ] );

    if ( name == "RGaussian" ) return make_and_test( name, degp, []( auto r ) { return r * exp( - r * r ); } );
    if ( name == "Gaussian"  ) return make_and_test( name, degp, []( auto r ) { return exp( - r * r ); } );
    if ( name == "Unit"      ) return make_and_test( name, degp, []( auto r ) { return 1;              } );
    if ( name == "R2"        ) return make_and_test( name, degp, []( auto r ) { return r * r;          } );

    std::cerr << argv[ 1 ] << " is not a known function type";
    return 1;
}
