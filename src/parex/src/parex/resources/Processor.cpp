#include "Processor.h"

namespace parex {

Processor::~Processor() {
}

void Processor::write_to_stream( std::ostream &os ) const {
    os << asimd_name();
    asimd_init( os, "_", " " );
}

bool Processor::cuda_device() const {
    return false;
}

} // namespace parex
