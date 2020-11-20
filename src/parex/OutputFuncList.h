#pragma once

#include <ostream>

namespace parex {

/**
*/
class OutputFuncList {
public:
    using        Destroy       = void( void * );
    using        WriteToStream = void( std::ostream &os, const void * );

    Destroy*     destroy;
};

} // namespace parex
