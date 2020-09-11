#include "lib/Poly2dApproximator.h"
#include <matplotlibcpp.h>

int main( int argc, char **argv ) {
    using TF = Poly2dApproximator::TF;
    Poly2dApproximator pd( 1e-3, 6 );

    int nr = 15, nt = 100;
    std::vector<TF> r, t;
    for( std::size_t ir = 0; ir < nr; ++ir )
        r.push_back( ir * 10.0 / ( nr - 1 ) );
    for( std::size_t it = 0; it < nt; ++it )
        t.push_back( it * 2 * M_PI / nt );

    pd.run_with_func( r, t, []( auto x, auto y ) {
        using fadbad::sqrt;
        return /*sqrt*/( x * x + y * y );
    } );

    P( pd );

    std::vector<double> px, py;
    for( std::size_t ir = 0; ir < nr; ++ir ) {
        for( std::size_t it = 0; it < nt; ++it ) {
            double x = double( r[ ir ] * cos( t[ it ] ) );
            double y = double( r[ ir ] * sin( t[ it ] ) );
            double v = double( pd.apply( x, y ) );
            px.push_back( x );
            py.push_back( y + 0.5 * v );
        }
    }
    matplotlibcpp::plot( px, py, "." );
    matplotlibcpp::show();
}
