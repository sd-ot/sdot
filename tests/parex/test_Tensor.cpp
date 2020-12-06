#include <parex/tasks/CompiledTask.h>
#include <parex/wrappers/Tensor.h>
#include <parex/utility/P.h>
#include "catch_main.h"

using namespace parex;

// a sample Task that will be executed on the CPU
class TestForceExecOnCpu : public CompiledTask {
public:
    using CompiledTask::CompiledTask;

    virtual void write_to_stream( std::ostream &os ) const override {
        os << "TestForceExecOnCpu";
    }

    virtual void get_src_content( Src &src, SrcSet &/*sw*/ ) override {
        src << "template<class T>\n";
        src << "auto " << called_func_name() << "( parex::TaskOut<T> &a ) {\n";
        src << "    return parex::TaskOut<int>( new int( 17 ) );\n";
        src << "}\n";

    }
};


TEST_CASE( "Tensor ctor", "[wrapper]" ) {
    CHECK( same_repr( Tensor( {{0,1,2},{3,4,5}} ), "0 1 2\n3 4 5" ) );
    CHECK( same_repr( Tensor( {
        { { 0, 1, 2 }, { 3,  4,  5 } },
        { { 6, 7, 8 }, { 9, 10, 11 } }
    } ), " 0  1  2\n 3  4  5\n\n 6  7  8\n 9 10 11" ) );
}

TEST_CASE( "Tensor with transfer", "[wrapper]" ) {
    Tensor t( {{0,1,2},{3,4,5}} );

    P( Scalar( new TestForceExecOnCpu( { t.task } ) ) );

}
