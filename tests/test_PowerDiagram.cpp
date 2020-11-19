#include <sdot/geometry/PowerDiagram.h>
#include <parex/containers/Tensor.h>
#include <parex/support/P.h>
#include <parex/Value.h>

using namespace parex;
using namespace sdot;

using TI = std::uint64_t;
using TF = double;

void test_fill( TI nb_diracs = 10 ) {
    Tensor<TF> *diracs = new Tensor<TF>( { nb_diracs, 3 } );


    PowerDiagram pd;
    pd.add_diracs( diracs );
}

int main() {
    test_fill();
}
