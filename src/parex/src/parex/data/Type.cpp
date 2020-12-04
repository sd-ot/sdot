#include "../plugins/Src.h"
#include "../utility/P.h"
#include "Type.h"

namespace parex {

Type::~Type() {
}

void Type::for_each_type_rec( const std::function<void(const Type *)> &cb ) const {
    cb( this );
}

void Type::add_needs_in( Src &src ) const {
    for_each_type_rec( [&](const Type *t ) {
        src.compilation_environment += t->compilation_environment;
    } );
}

} // namespace parex
