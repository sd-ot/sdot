#pragma once

#include <limits>

namespace sdot {

template<class T,class S,int d>
struct ZGridUpdateParms {
    using  Pt               = Point<T,d>;

    S      approx_nb_diracs = 0;                               ///<
    Pt     incl_min_point   = - std::numeric_limits<T>::max(); ///<
    Pt     incl_max_point   = + std::numeric_limits<T>::max(); ///<
    bool   new_positions    = true;                            ///<
    bool   new_weights      = true;                            ///<
};

}
