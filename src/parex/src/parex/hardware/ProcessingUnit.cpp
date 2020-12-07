#include "ProcessingUnit.h"

namespace parex {

ProcessingUnit::~ProcessingUnit() {
}

void ProcessingUnit::write_to_stream( std::ostream &os ) const {
    os << asimd_name();
    asimd_init( os, "_", " " );
}

bool ProcessingUnit::cuda_device() const {
    return false;
}

} // namespace parex
