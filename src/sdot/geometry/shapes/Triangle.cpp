#include "Triangle.h"

namespace sdot {

Shape &triangle() {
   Shape res;
   if ( res.nb_points == 0 ) {
       res.nb_points = 3;
       res.nb_faces = 3;


   }
   return res;
}


} // namespace sdot
