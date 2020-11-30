#include "SrcTask.h"
#include <ostream>

SrcTask::SrcTask( Type *type, void *data, bool own ) {
    output_is_owned = own;
    output_type = type;
    output_data = data;
}

void SrcTask::write_to_stream( std::ostream &os ) const {
    os << "src";
}
