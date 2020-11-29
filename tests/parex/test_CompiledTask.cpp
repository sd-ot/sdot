#include <parex/CompiledIncludeTask.h>
#include <parex/CompiledLambdaTask.h>
#include <parex/P.h>

int main() {
    Rc<CompiledLambdaTask> clt = new CompiledLambdaTask( []( Src &src, SrcSet &, const std::vector<Rc<Task>> & ) {
        src << "TaskOut<int> smurf() { return new int( 12 ); }";
    }, {}, "smurf" );
    clt->exec();
    P( *clt->output_type );

    Rc<CompiledIncludeTask> cit = new CompiledIncludeTask( "parex/kernels/to_string.h", { clt } );
    cit->exec();
    P( *cit->output_type );
}
