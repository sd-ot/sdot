#include <sdot/geometry/PowerDiagram.h>
#include <parex/containers/Tensor.h>
#include <parex/support/P.h>
#include <parex/Value.h>

using namespace parex;
using namespace sdot;

using TI = std::uint64_t;
using TF = double;

Tensor<TF> *random_diracs( TI nb_diracs = 10 ) {
    Tensor<TF> *diracs = new Tensor<TF>( { nb_diracs, 3 } );
    for( TI i = 0; i < nb_diracs; ++i ) {
        diracs->ptr( 0 )[ i ] = 1.0 * rand() / RAND_MAX;
        diracs->ptr( 1 )[ i ] = 1.0 * rand() / RAND_MAX;
        diracs->ptr( 2 )[ i ] = 0.0;
    }

    return diracs;
}

void test_fill() {
    PowerDiagram pd( 2 );
    for( TI i = 0; i < 4; ++i )
        pd.add_diracs( random_diracs() );

    pd.for_each_cell( [&]( const Value &cells ) {
        P( cells );
    } );
}

int main() {
    test_fill();
}


