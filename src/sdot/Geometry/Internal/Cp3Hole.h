#pragma once

#include "../../Geometry/Point3.h"

namespace sdot {

/**
*/
template<class Pc>
struct Cp3Hole {
    using    TF            = typename Pc::TF; ///< floating point type
    using    CI            = typename Pc::CI; ///< cut info
    using    Pt            = Point3<TF>;      ///< 3D point

    Cp3Hole *prev_in_pool; ///<
    Cp3Hole *next_in_pool; ///<
    CI       cut_id;       ///< provided by the user (as argument of the cut function)
    Pt       cut_O;        ///< a point in the cut plane
    Pt       cut_N;        ///< normal, oriented toward exterior, i.e. the removed part (if the face is actually plane)
};

} // namespace sdot

