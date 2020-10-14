#pragma once

#include "Rational.h"
#include <vector>

namespace sdot {

/**
*/
struct RecursivePolytopConnectivity {
    using           Rc                          = RecursivePolytopConnectivity;
    using           TI                          = std::size_t;
    using           TF                          = Rational;

    /**/            RecursivePolytopConnectivity( TI nvi );
    void            write_to_stream             ( std::ostream &os ) const;

    std::vector<Rc> sub_polytops;
    std::vector<TI> nodes;
    TI              nvi;
};

} // namespace sdot
