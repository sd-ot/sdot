#pragma once

#include <vector>
#include <string>

namespace sdot {

/**
*/
class SymbolicElementaryPolytop {
public:
    using           TI                       = std::size_t;

    /**/            SymbolicElementaryPolytop( const std::string &name );

    std::string     vtk_output               () const;
    unsigned        nb_nodes                 () const;
    unsigned        nb_faces                 () const;

    std::vector<TI> nodes;
    std::string     name;
};

} // namespace sdot
