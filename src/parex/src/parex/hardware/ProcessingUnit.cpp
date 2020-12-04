#include "ProcessingUnit.h"

namespace parex {
namespace hardware_information {

ProcessingUnit::~ProcessingUnit() {
}

void ProcessingUnit::write_to_stream( std::ostream &os ) const {
    os << asimd_name();
    asimd_init( os, "_", " " );
}

} // namespace hardware_information
} // namespace parex
