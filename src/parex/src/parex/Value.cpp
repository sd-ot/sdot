#include "Tasks/GenericArithmeticOperation.h"
#include "CompiledIncludeTask.h"
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
    Rc<Task> ts = to_string( std::numeric_limits<double>::max() );
    scheduler.append( ts );
    scheduler.run();

    os << *reinterpret_cast<const std::string *>( ts->output_data );
}

Value Value::operator+( const Value &that ) const { return (Task *)new GenericArithmeticOperation( "+", { task, that.task } ); }
Value Value::operator-( const Value &that ) const { return (Task *)new GenericArithmeticOperation( "-", { task, that.task } ); }
Value Value::operator*( const Value &that ) const { return (Task *)new GenericArithmeticOperation( "*", { task, that.task } ); }
Value Value::operator/( const Value &that ) const { return (Task *)new GenericArithmeticOperation( "/", { task, that.task } ); }

Value &Value::operator+=( const Value &that ) { task = new GenericArithmeticOperation( "+", { task, that.task } ); return *this; }
Value &Value::operator-=( const Value &that ) { task = new GenericArithmeticOperation( "-", { task, that.task } ); return *this; }
Value &Value::operator*=( const Value &that ) { task = new GenericArithmeticOperation( "*", { task, that.task } ); return *this; }
Value &Value::operator/=( const Value &that ) { task = new GenericArithmeticOperation( "/", { task, that.task } ); return *this; }

Rc<Task> Value::to_string( double priority ) const {
    return static_cast<Task *>( new CompiledIncludeTask( "parex/kernels/to_string.h", { task }, {}, priority ) );
}

Rc<Task> Value::conv_to( Type *type ) const {
    return static_cast<Task *>( new CompiledIncludeTask( "parex/kernels/conv_to.h", {
        static_cast<Task *>( new SrcTask( type, nullptr, false ) ),
        task
    }, {} ) );
}

Rc<Task> Value::conv_to( std::string type_name ) const {
    return conv_to( task->type_factory_virtual()( type_name ) );
}

