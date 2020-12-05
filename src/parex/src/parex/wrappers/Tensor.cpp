#include "../tasks/GenericArithmeticOperation.h"
#include "../tasks/CompiledIncludeTask.h"
#include "Tensor.h"

namespace parex {

Tensor::Tensor( Task *t ) : TaskWrapper( t ) {}

Tensor Tensor::operator+( const Tensor &that ) const { return new GenericArithmeticOperation( "+", { task, that.task } ); }
Tensor Tensor::operator-( const Tensor &that ) const { return new GenericArithmeticOperation( "-", { task, that.task } ); }
Tensor Tensor::operator*( const Tensor &that ) const { return new GenericArithmeticOperation( "*", { task, that.task } ); }
Tensor Tensor::operator/( const Tensor &that ) const { return new GenericArithmeticOperation( "/", { task, that.task } ); }

Tensor &Tensor::operator+=( const Tensor &that ) { task = new GenericArithmeticOperation( "+", { task, that.task } ); return *this; }
Tensor &Tensor::operator-=( const Tensor &that ) { task = new GenericArithmeticOperation( "-", { task, that.task } ); return *this; }
Tensor &Tensor::operator*=( const Tensor &that ) { task = new GenericArithmeticOperation( "*", { task, that.task } ); return *this; }
Tensor &Tensor::operator/=( const Tensor &that ) { task = new GenericArithmeticOperation( "/", { task, that.task } ); return *this; }

} // namespace parex
