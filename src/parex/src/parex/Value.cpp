#include "Tasks/GenericArithmeticOperation.h"
#include "Tasks/CompiledTaskWithInclude.h"
#include "Scheduler.h"
#include "Value.h"
#include "P.h"

Value::Value( const Rc<Task> &task ) : task( task ) {
}

Value::Value( Rc<Task> &&task ) : task( std::move( task ) ) {
}

Value::Value( Task *task ) : task( task ) {
}

Value::Value( const char *value ) : Value( new std::string( value ), /*owned*/ true ) {
}

void Value::write_to_stream( std::ostream &os ) const {
    Rc<CompiledTaskWithInclude> ts = new CompiledTaskWithInclude( "parex/kernels/to_string.h", { task }, 1 );
    scheduler.append( ts );
    scheduler.run();

    os << *reinterpret_cast<const std::string *>( ts->output_data() );
}

Value Value::operator+( const Value &that ) const { return (Task *)new GenericArithmeticOperation( "+", { task, that.task } ); }
Value Value::operator-( const Value &that ) const { return (Task *)new GenericArithmeticOperation( "-", { task, that.task } ); }
Value Value::operator*( const Value &that ) const { return (Task *)new GenericArithmeticOperation( "*", { task, that.task } ); }
Value Value::operator/( const Value &that ) const { return (Task *)new GenericArithmeticOperation( "/", { task, that.task } ); }

Value &Value::operator+=( const Value &that ) { task = new GenericArithmeticOperation( "+", { task, that.task } ); return *this; }
Value &Value::operator-=( const Value &that ) { task = new GenericArithmeticOperation( "-", { task, that.task } ); return *this; }
Value &Value::operator*=( const Value &that ) { task = new GenericArithmeticOperation( "*", { task, that.task } ); return *this; }
Value &Value::operator/=( const Value &that ) { task = new GenericArithmeticOperation( "/", { task, that.task } ); return *this; }
