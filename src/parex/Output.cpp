#include "KernelCode.h"
#include "Output.h"

namespace parex {

void Output::destroy() {
    if ( own && data && ! type.empty() ) {
        KernelCode::Func f = kernel_code.func( { "destroy" }, { type } );
        f( nullptr, &data );
    }
}

} // namespace parex
