#include "../tasks/GenericArithmeticOperation.h"
#include "../tasks/CompiledIncludeTask.h"
#include "String.h"

namespace parex {

String::String( const std::string &str ) : String( Task::new_src_from_ptr( new std::string( str ), true ) ) {
}

String::String( const char *str ) : String( std::string( str ) ) {
}

String::String( Task *t ) : TaskWrapper( t ) {
}

Scalar String::size() const {
    return new CompiledIncludeTask( "parex/kernels/size.h", { task } );
}

String String::operator+( const String &that ) const { return new GenericArithmeticOperation( "+", { task, that.task } ); }

String &String::operator+=( const String &that ) { task = new GenericArithmeticOperation( "+", { task, that.task } ); return *this; }

} // namespace parex
