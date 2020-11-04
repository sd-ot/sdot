#include "../src/sdot/ParEx/Scheduler.h"
#include "../src/sdot/ParEx/Kernel.h"
#include "../src/sdot/ParEx/Value.h"
#include "../src/sdot/support/P.h"

using namespace parex;
using TF = double;
using TI = int;

template<class TF=double>
Value uniform_random_vec( std::size_t size, TF min = 0.0, TF max = 1.0 ) {
    return { Kernel{ "uniform_noise_vec" }, size, min, max };
}

Value min_max( const Value &value ) {
    return { Kernel{ "min_max" }, { value } };
}

void display( const Value &value, std::ostream &os = std::cout ) {
    scheduler << Value{ Kernel{ "write_to_stream" }, { &os, value } };
}

int main() {
    //    Scheduler sch;

    //    TI dim = 2;
    //    TI nb_diracs = 200;

    //    std::vector<Node> pos;
    //    for( TI d = 0; d < dim; ++d )
    //        pos.push_back( uniform_random_vec( nb_diracs, 0.0, 1.0 ) );

    //    Node m_m = { { "min_max" }, pos };
    //    sch << Node{ { "display" }, std::cout, m_m };
    Value v( 17 );
    P( v );
}
