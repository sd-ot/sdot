#pragma once

#include "TetraAssembly.h"

/*
*/
class Volume {
public:
    using           Pt = TetraAssembly::Pt;
    using           TI = TetraAssembly::TI;
    using           TF = TetraAssembly::TF;

    TetraAssembly   tetra_assembly;
    TF              volume;
    std::vector<Pt> points;
    std::vector<TI> nodes;
};

