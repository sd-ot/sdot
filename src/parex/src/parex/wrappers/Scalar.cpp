#include "../tasks/GenericArithmeticOperation.h"
#include "../tasks/CompiledIncludeTask.h"
#include "../utility/TODO.h"
#include "Scalar.h"

namespace parex {

Scalar::Scalar( Task *t ) : TaskWrapper( t ) {}
Scalar::Scalar() { TODO; }

Scalar Scalar::operator+( const Scalar &that ) const { return new GenericArithmeticOperation( "+", { task, that.task } ); }
Scalar Scalar::operator-( const Scalar &that ) const { return new GenericArithmeticOperation( "-", { task, that.task } ); }
Scalar Scalar::operator*( const Scalar &that ) const { return new GenericArithmeticOperation( "*", { task, that.task } ); }
Scalar Scalar::operator/( const Scalar &that ) const { return new GenericArithmeticOperation( "/", { task, that.task } ); }

Scalar &Scalar::operator+=( const Scalar &that ) { task = new GenericArithmeticOperation( "+", { task, that.task } ); return *this; }
Scalar &Scalar::operator-=( const Scalar &that ) { task = new GenericArithmeticOperation( "-", { task, that.task } ); return *this; }
Scalar &Scalar::operator*=( const Scalar &that ) { task = new GenericArithmeticOperation( "*", { task, that.task } ); return *this; }
Scalar &Scalar::operator/=( const Scalar &that ) { task = new GenericArithmeticOperation( "/", { task, that.task } ); return *this; }

} // namespace parex
