#pragma once

#include <parex/containers/xtensor.h>
#include <vector>

/**
*/
class ElementaryPolytopInfo {
public:
    using             TI                   = std::size_t;

    /**/              ElementaryPolytopInfo( const std::string &name );
    TI                nb_nodes             () const;

    xt::xtensor<TI,1> nodes;
    std::string       name;
};

