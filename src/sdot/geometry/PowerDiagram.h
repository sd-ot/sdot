#pragma once

#include <parex/Value.h>

namespace sdot {

/**
*/
class PowerDiagram {
public:
    /**/ PowerDiagram();

    void add_diracs  ( const parex::Value &diracs );
};

} // namespace sdot
