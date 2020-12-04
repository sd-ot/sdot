#include "../tasks/GenericArithmeticOperation.h"
#include "../tasks/CompiledIncludeTask.h"
#include "Scalar.h"

namespace parex {

Scalar::Scalar( const char *str ) : TaskWrapper( new SrcTask( Task::type_factory( TypeInfo<std::string>::name() ), new std::string( str ), true ) ) {}

Scalar Scalar::operator+( const Scalar &that ) const { return (Task *)new GenericArithmeticOperation( "+", { task, that.task } ); }
Scalar Scalar::operator-( const Scalar &that ) const { return (Task *)new GenericArithmeticOperation( "-", { task, that.task } ); }
Scalar Scalar::operator*( const Scalar &that ) const { return (Task *)new GenericArithmeticOperation( "*", { task, that.task } ); }
Scalar Scalar::operator/( const Scalar &that ) const { return (Task *)new GenericArithmeticOperation( "/", { task, that.task } ); }

Scalar &Scalar::operator+=( const Scalar &that ) { task = new GenericArithmeticOperation( "+", { task, that.task } ); return *this; }
Scalar &Scalar::operator-=( const Scalar &that ) { task = new GenericArithmeticOperation( "-", { task, that.task } ); return *this; }
Scalar &Scalar::operator*=( const Scalar &that ) { task = new GenericArithmeticOperation( "*", { task, that.task } ); return *this; }
Scalar &Scalar::operator/=( const Scalar &that ) { task = new GenericArithmeticOperation( "/", { task, that.task } ); return *this; }

} // namespace parex
