#include "../support/generic_ostream_output.h"
#include "Tensor.h"

namespace parex {

template<class T,class A,class TI>
void Tensor<T,A,TI>::write_to_stream( std::ostream &os ) const {
    os << data;
}


} // namespace parex
