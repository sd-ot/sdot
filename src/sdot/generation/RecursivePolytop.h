#pragma once

#include "RecursivePolytopConnectivity.h"

namespace sdot {

/**
*/
struct RecursivePolytop {
    using           Connectivity    = RecursivePolytopConnectivity;
    using           Pt              = std::vector<Rational>;
    using           Rp              = RecursivePolytop;
    using           TI              = std::size_t;
    using           TF              = Rational;

    /**/            RecursivePolytop( TI nvi );

    static Rp       convex_hull     ( const std::vector<Pt> &points );

    void            write_to_stream ( std::ostream &os ) const;
    TI              nb_faces        () const;

    Connectivity    connectivity;
    std::vector<Pt> points;
};

} // namespace sdot
