#include "Scheduler.h"
#include "Kernel.h"
#include "Value.h"

namespace parex {

Value::Value( TaskRef &&ref ) : ref( std::move( ref ) ) {
}

Value::Value( Task *task ) : ref( task ) {
}

void Value::write_to_stream( std::ostream &os ) const {
    scheduler << Task::call( new Kernel{ "write_to_stream" }, { Task::owning( &os ), ref } );
    scheduler.run();
}

} // namespace parex
