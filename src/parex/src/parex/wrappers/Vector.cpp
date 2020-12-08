#include "../tasks/CompiledIncludeTask.h"
#include "Vector.h"

namespace parex {

Vector::Vector( Task *t ) : TaskWrapper( t ) {
}

Scalar Vector::size() const {
    return new CompiledIncludeTask( "parex/kernels/size.h", { task } );
}

} // namespace parex
