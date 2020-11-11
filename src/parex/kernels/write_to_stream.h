#include <iostream>

template<class T>
void write_to_stream( std::ostream &os, const T &value ) {
    os << value;
}

