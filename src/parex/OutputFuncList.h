#pragma once

#include <ostream>

namespace parex {

/**
*/
class OutputFuncList {
public:
    using          WriteToStream = void( std::ostream &os, const void *data );
    using          Destroy       = void( std::size_t *ref_count, void *data );

    WriteToStream *write_to_stream;
    Destroy       *destroy;
};

} // namespace parex
