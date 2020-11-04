#include "../../../kernels/write_to_stream.h"

extern "C" void kernel_wrapper( void **data ) {
    write_to_stream(
        *reinterpret_cast<std::ostream*>( data[ 0 ] ),
        *reinterpret_cast<std::int32_t*>( data[ 1 ] )
    );
}
