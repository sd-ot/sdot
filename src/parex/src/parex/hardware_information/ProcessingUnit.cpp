#include "ProcessingUnit.h"

namespace asimd {
namespace hardware_information {

ProcessingUnit::~ProcessingUnit() {
}

void ProcessingUnit::write_to_stream( std::ostream &os ) const {
    os << asimd_name();
}

} // namespace hardware_information
} // namespace asimd
