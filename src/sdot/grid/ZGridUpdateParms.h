#pragma once

#include <limits>
#include <cmath>

namespace sdot {

/*
  The first traversal of each update can be avoided if approx_nb_diracs, hist_min_point and hist_max_point are defined.
*/
template<class T,class S,int d>
struct ZGridUpdateParms {
    using  Pt                         = Point<T,d>;

    bool   positions_may_have_changed = true;                                ///<
    bool   weights_may_have_changed   = true;                                ///<
    bool   may_not_fit_in_memory      = true;                                ///<
    S      approx_nb_diracs           = 0;                                   ///< if != 0, used during the first traversal to get hist size
    Pt     hist_min_point             = std::numeric_limits<T>::quiet_NaN(); ///< if defined, it is used to make an histogram of dirac/zcoord
    Pt     hist_max_point             = std::numeric_limits<T>::quiet_NaN(); ///< if defined, it is used to make an histogram of dirac/zcoord
    T      hist_ratio                 = 1.0;                                 ///< histogram size is equal to nb_diracs (or approx) * hist_ratio
};

}
