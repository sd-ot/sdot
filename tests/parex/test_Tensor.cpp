#include <parex/containers/CudaAllocator.h>
#include <parex/tasks/CompiledTask.h>
#include <parex/tasks/Scheduler.h>
#include <parex/hardware/HwGraph.h>
#include <parex/wrappers/Tensor.h>
#include <parex/utility/P.h>
#include "catch_main.h"

using namespace parex;

// a sample Task that will be executed on the CPU
class TestForceExecOnCpu : public CompiledTask {
public:
    TestForceExecOnCpu( std::vector<Rc<Task>> &&children, hardware_information::Memory *mem ) : CompiledTask( "TestForceExecOnCpu", std::move( children ) ), mem( mem ) {
    }

    virtual void prepare() override {
        check_output_alloc( 0, mem );
    }

    virtual void check_output_alloc( std::size_t num_child, hardware_information::Memory *mem ) {
        Type *type = children[ num_child ]->output.type;
        VecUnique<hardware_information::Memory *> memories;
        type->get_memories( memories, children[ num_child ]->output.data );
        if ( std::find_if( memories.begin(), memories.end(), [&]( hardware_information::Memory *m ) { return m != mem; } ) != memories.end() ) {
            insert_child( num_child, type->conv_alloc_task( move_child( num_child ), mem ) );
            computed = false;
        }
    }

    virtual void get_src_content( Src &src, SrcSet &/*sw*/ ) override {
        src << "template<class T,int D,class A>\n";
        src << "parex::TaskOut<std::string> " << called_func_name() << "( parex::TaskOut<parex::gtensor<T,D,A>> &a ) {\n";
        src << "    return new std::string( parex::TypeInfo<A>::name() );\n";
        src << "}\n";
    }

    hardware_information::Memory *mem;
};


TEST_CASE( "Tensor ctor", "[wrapper]" ) {
    CHECK( same_repr( Tensor( {
        { 0, 1, 2 },
        { 3, 4, 5 }
    } ), "0 1 2\n3 4 5" ) );

    CHECK( same_repr( Tensor( {
        { { 0, 1, 2 }, { 3,  4,  5 } },
        { { 6, 7, 8 }, { 9, 10, 11 } }
    } ), " 0  1  2\n 3  4  5\n\n 6  7  8\n 9 10 11" ) );
}

TEST_CASE( "Tensor to allocator", "[wrapper]" ) {
    Tensor inp( { { 0, 1, 2 }, { 3, 4, 5 } } );
    scheduler.log = true;
    for( const auto &mem : hw_graph()->memories ) {
        P( Scalar( new TestForceExecOnCpu( { inp.task }, mem.get() ) ) );
    }
}
