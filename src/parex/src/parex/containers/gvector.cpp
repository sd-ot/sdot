#include "../data/TypeFactoryRegister.h"
#include "../plugins/Src.h"
#include "gvector.h"

namespace parex {

struct Type_gvector : Type {
    virtual void gen_func_get_memories( Src &src, SrcSet &/*sw*/ ) const override {
        src << "template<class T,class A> void get_memories( parex::VecUnique<parex::Memory *> &memories, const parex::HwGraph *hw_graph, const parex::gvector<T,A> *data ) {\n";
        src << "    memories << data->allocator()->memory();\n";
        src << "}\n";
    }
};


static TypeFactoryRegister _0( { "parex::gvector" }, []( TypeFactory &tf, const std::vector<std::string> &parameters ) {
    Type *res = new Type_gvector;
    res->compilation_environment.includes << "<parex/containers/gvector.h>";
    res->sub_types = { tf( parameters[ 0 ] ), tf( parameters[ 1 ] ) };
    return res;
} );

} // namespace parex
