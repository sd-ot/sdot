#include "../data/TypeFactoryRegister.h"
#include "../plugins/Src.h"
#include "gtensor.h"

namespace parex {

struct Type_gtensor : Type {
    virtual void gen_func_get_memories( Src &src, SrcSet &/*sw*/ ) const override {
        // src.compilation_environment.includes << "<parex/hardware/default_CpuAllocator.h>";
        // src << "    memories << &default_CpuAllocator.memory;\n";
        src << "template<class T,int D,class A> void get_memories( parex::VecUnique<parex::Memory *> &memories, const parex::HwGraph *hw_graph, const parex::gtensor<T,D,A> *data ) {\n";
        src << "    memories << data->allocator()->memory();\n";
        src << "}\n";
    }
};


static TypeFactoryRegister _0( { "parex::gtensor" }, []( TypeFactory &tf, const std::vector<std::string> &parameters ) {
    Type *res = new Type_gtensor;
    res->compilation_environment.includes << "<parex/containers/gtensor.h>";
    res->sub_types = { tf( parameters[ 0 ] ), tf( parameters[ 2 ] ) };
    return res;
} );

} // namespace parex
