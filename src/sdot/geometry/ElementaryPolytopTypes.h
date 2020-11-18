#pragma once

#include <parex/Value.h>

namespace sdot {


/**
*/
class ElementaryPolytopTypes {
public:
    /**/               ElementaryPolytopTypes( const parex::Value &N_dim, const std::string &shape_names );
    /**/               ElementaryPolytopTypes( int dim );

    static std::string shape_names_for       ( int dim );

    parex::TaskRef     operations;
    parex::TaskRef     dim;
};

} // namespace sdot
