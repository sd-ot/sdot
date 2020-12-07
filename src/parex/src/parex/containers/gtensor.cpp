#include "../data/TypeFactoryRegister.h"
#include "gtensor.h"

namespace parex {

struct Type_gtensor : Type {

};

static TypeFactoryRegister _0( { "parex::gtensor" }, []( TypeFactory &tf, const std::vector<std::string> &parameters ) {
    Type_gtensor *res = new Type_gtensor;
    res->compilation_environment.includes << "<parex/containers/gtensor.h>";
    res->sub_types = { tf( parameters[ 0 ] ), tf( parameters[ 2 ] ) };
    return res;
} );

} // namespace parex
