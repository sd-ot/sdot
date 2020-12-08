#include "../tasks/Task.h"
#include "Memory.h"

namespace parex {

Memory::Memory() : amount( I( -1 ) ), used( 0 ) {
}

Rc<Task> Memory::allocator_as_task() {
    return Task::new_src( Task::type_factory( allocator_type() ), allocator_data(), false );
}

void Memory::register_link( const ProcLink &link ) {
    bw_to_pu_links[ link.bandwidth ].push_back( link );
    pu_to_pu_link[ link.processing_unit ] = link;
}

} // namespace parex
