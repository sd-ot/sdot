#include "Memory.h"

namespace parex {

Memory::Memory() : amount( I( -1 ) ), used( 0 ) {
}

Memory::~Memory() {
}

void Memory::register_link( const ProcLink &link ) {
    bw_to_pu_links[ link.bandwidth ].push_back( link );
    pu_to_pu_link[ link.processing_unit ] = link;
}

} // namespace parex
