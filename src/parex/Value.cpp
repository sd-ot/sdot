#include "Scheduler.h"
#include "Kernel.h"
#include "Value.h"

#include "support/P.h"

namespace parex {

Value::Value( TaskRef &&ref ) : ref( std::move( ref ) ) {
}

Value::Value( Task *task ) : ref( task ) {
}

void Value::write_to_stream( std::ostream &os ) const {
    scheduler << Task::call( "write_to_stream", {}, { Task::ref_on( &os, false ), ref } );
    scheduler.run();
}

Value Value::operator+( const Value &that ) const { return Task::call_r( Kernel::with_task_as_arg( "gen_op(+)" ), { ref, that.ref } ); }
Value Value::operator-( const Value &that ) const { return Task::call_r( Kernel::with_task_as_arg( "gen_op(-)" ), { ref, that.ref } ); }
Value Value::operator*( const Value &that ) const { return Task::call_r( Kernel::with_task_as_arg( "gen_op(*)" ), { ref, that.ref } ); }
Value Value::operator/( const Value &that ) const { return Task::call_r( Kernel::with_task_as_arg( "gen_op(/)" ), { ref, that.ref } ); }

Value &Value::operator+=( const Value &that ) { ref = Task::call_r( Kernel::with_task_as_arg( "gen_op(+)" ), { ref, that.ref } ); return *this; }
Value &Value::operator-=( const Value &that ) { ref = Task::call_r( Kernel::with_task_as_arg( "gen_op(-)" ), { ref, that.ref } ); return *this; }
Value &Value::operator*=( const Value &that ) { ref = Task::call_r( Kernel::with_task_as_arg( "gen_op(*)" ), { ref, that.ref } ); return *this; }
Value &Value::operator/=( const Value &that ) { ref = Task::call_r( Kernel::with_task_as_arg( "gen_op(/)" ), { ref, that.ref } ); return *this; }

} // namespace parex
