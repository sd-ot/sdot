#include "SrcTask.h"
#include <ostream>

namespace parex {

SrcTask::SrcTask( Type *type, void *data, bool own ) {
    output.type = type;
    output.data = data;
    output.own = own;
}

void SrcTask::write_to_stream( std::ostream &os ) const {
    os << "src";
}

} // namespace parex
