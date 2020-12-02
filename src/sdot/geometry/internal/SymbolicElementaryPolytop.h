#pragma once

#include <parex/containers/xtensor.h>
#include <string>

/**
*/
class SymbolicElementaryPolytop {
public:
    using             TI                       = std::size_t;

    /**/              SymbolicElementaryPolytop( const std::string &name );

    std::string       vtk_output               () const;
    unsigned          nb_nodes                 () const;
    unsigned          nb_faces                 () const;

    xt::xtensor<TI,1> nodes;
    std::string       name;
};

