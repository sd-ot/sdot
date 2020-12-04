#include "Memory.h"

namespace parex {
namespace hardware_information {

Memory::~Memory() {
}

void Memory::register_link( const PULink &link ) {
    bw_to_pu_links[ link.bandwidth ].push_back( link );
    pu_to_pu_link[ link.processing_unit ] = link;
}

} // namespace hardware_information
} // namespace parex
