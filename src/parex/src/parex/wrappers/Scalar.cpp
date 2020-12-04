#include "../task_graphs/GenericArithmeticOperation.h"
#include "../task_graphs/CompiledIncludeTask.h"
#include "Scalar.h"

namespace parex {

Scalar::Scalar( const char *str ) : Scalar( new SrcTask( Task::type_factory( type_name<std::string>() ), new std::string( str ), true ) ) {}

Scalar Scalar::operator+( const Scalar &that ) const { return (Task *)new GenericArithmeticOperation( "+", { task, that.task } ); }
Scalar Scalar::operator-( const Scalar &that ) const { return (Task *)new GenericArithmeticOperation( "-", { task, that.task } ); }
Scalar Scalar::operator*( const Scalar &that ) const { return (Task *)new GenericArithmeticOperation( "*", { task, that.task } ); }
Scalar Scalar::operator/( const Scalar &that ) const { return (Task *)new GenericArithmeticOperation( "/", { task, that.task } ); }

Scalar &Scalar::operator+=( const Scalar &that ) { task = new GenericArithmeticOperation( "+", { task, that.task } ); return *this; }
Scalar &Scalar::operator-=( const Scalar &that ) { task = new GenericArithmeticOperation( "-", { task, that.task } ); return *this; }
Scalar &Scalar::operator*=( const Scalar &that ) { task = new GenericArithmeticOperation( "*", { task, that.task } ); return *this; }
Scalar &Scalar::operator/=( const Scalar &that ) { task = new GenericArithmeticOperation( "/", { task, that.task } ); return *this; }

} // namespace parex
