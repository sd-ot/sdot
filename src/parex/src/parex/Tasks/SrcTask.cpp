#include "SrcTask.h"

SrcTask::SrcTask( Type *type, void *data, bool own ) : type( type ), data( data ), own( own ) {
}

SrcTask::~SrcTask() {
    if ( type )
        type->destroy( data );
}

void SrcTask::write_to_stream( std::ostream &os ) const {
    os << "src";
}

bool SrcTask::is_computed() const {
    return true;
}

Type* SrcTask::output_type() const {
    return type;
}

void* SrcTask::output_data() const {
    return data;
}
