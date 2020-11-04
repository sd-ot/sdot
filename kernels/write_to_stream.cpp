#include <iostream>

template<class T>
void kernel( std::ostream &os, const T &value ) {
    os << value;
}

extern "C" void kernel_wrapper( void **data ) {
    kernel(
        *reinterpret_cast<std::ostream *>( data[ 0 ] ),
        *reinterpret_cast<int *>( data[ 1 ] )
    );
}
