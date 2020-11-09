#include "../src/parex/containers/Tensor.h"
#include "../src/parex/KernelCode.h"
#include "../src/parex/Scheduler.h"
#include "../src/parex/Kernel.h"

#include "../src/parex/support/P.h"

using namespace parex;
using TF = double;
using TI = int;

Value random_vec( const Value &size, const Value &min = 0.0, const Value &max = 1.0 ) {
    return Task::call_r( new Kernel{ "random_vec" }, { size.ref, min.ref, max.ref } );
}

void min_max( Value &min, Value &max, const Value &container ) {
    Task::call( new Kernel{ "min_max" }, { &min.ref, &max.ref }, { container.ref } );
}

//Value min_max( const Value &value ) {
//    return { new Kernel{ "min_max" }, { value } };
//}

void display( const Value &value, std::ostream &os = std::cout ) {
    scheduler << Task::call( new Kernel{ "write_to_stream" }, {}, { Task::ref_on( &os, false ), value.ref } );
}

int main() {
    kernel_code.add_include_dir( SDOT_DIR "/src/sdot/kernels" );
    //    Scheduler sch;

    //    TI dim = 2;
    //    TI nb_diracs = 200;

    //    std::vector<Node> pos;
    //    for( TI d = 0; d < dim; ++d )
    //        pos.push_back( uniform_random_vec( nb_diracs, 0.0, 1.0 ) );

    //    Node m_m = { { "min_max" }, pos };
    //    sch << Node{ { "display" }, std::cout, m_m };
    //    Value v( 17 );
    //    P( v );
    //    P( min_max( Tensor<int>( { 4, 2 }, { 0, 1, 2, 3, 4, 5, 6, 7 } ) ) );

    // Tensor
    Value v( 17 );
//    v += 13;
    P( v + 13 );
}
