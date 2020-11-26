#include "KernelWithCodeInInclude.h"
#include "Scheduler.h"
#include "Value.h"
//#include "support/P.h"

Value::Value( const RcPtr<Task> &task ) : task( task ) {
}

Value::Value( RcPtr<Task> &&task ) : task( std::move( task ) ) {
}

Value::Value( Task *task ) : task( task ) {
}

Value::Value( const char *value ) : task( Task::from_data_ptr( new std::string( value ) ) ) {
}

void Value::write_to_stream( std::ostream &os ) const {
    static KernelWithCodeInInclude kernel( "parex/kernels/to_string.h" );
    RcPtr<Task> ts = Task::call( &kernel, { task } );
    ts->priority = 1;

    scheduler.append_target( ts );
    scheduler.run();

    os << reinterpret_cast<const std::string *>( ts->output->variants[ 0 ].data );
}

//Value Value::operator+( const Value &that ) const { return Task::call_r( Kernel::with_task_as_arg( "gen_op(+)" ), { ref, that.ref } ); }
//Value Value::operator-( const Value &that ) const { return Task::call_r( Kernel::with_task_as_arg( "gen_op(-)" ), { ref, that.ref } ); }
//Value Value::operator*( const Value &that ) const { return Task::call_r( Kernel::with_task_as_arg( "gen_op(*)" ), { ref, that.ref } ); }
//Value Value::operator/( const Value &that ) const { return Task::call_r( Kernel::with_task_as_arg( "gen_op(/)" ), { ref, that.ref } ); }

//Value &Value::operator+=( const Value &that ) { ref = Task::call_r( Kernel::with_task_as_arg( "gen_op(+)" ), { ref, that.ref } ); return *this; }
//Value &Value::operator-=( const Value &that ) { ref = Task::call_r( Kernel::with_task_as_arg( "gen_op(-)" ), { ref, that.ref } ); return *this; }
//Value &Value::operator*=( const Value &that ) { ref = Task::call_r( Kernel::with_task_as_arg( "gen_op(*)" ), { ref, that.ref } ); return *this; }
//Value &Value::operator/=( const Value &that ) { ref = Task::call_r( Kernel::with_task_as_arg( "gen_op(/)" ), { ref, that.ref } ); return *this; }
