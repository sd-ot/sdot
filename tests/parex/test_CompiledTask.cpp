#include <parex/CompiledLambdaTask.h>
#include <parex/P.h>

int main() {
    CompiledLambdaTask clt( []( Src &src, SrcSet &, const std::vector<Rc<Task>> & ) {
        src << "TaskOut<int> smurf() { return new int( 12 ); }";
    }, {}, "smurf" );

    clt.exec();
    P( *clt.output_type );
}
