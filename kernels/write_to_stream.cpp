#include "../src/sdot/ParEx/Value.h"
#include <iostream>

extern "C" void kernel( const parex::Value *value_data, std::size_t value_size ) {
    std::cout << "pouet:";
    for( std::size_t i = 0; i < value_size; ++i )
        value_data[ i ].write_to_stream( std::cout << " " );
    std::cout << "\n";
}
