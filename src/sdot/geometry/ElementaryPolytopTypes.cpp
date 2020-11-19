#include "ElementaryPolytopTypes.h"
#include <parex/support/TODO.h>
#include <parex/TaskRef.h>

namespace sdot {

ElementaryPolytopTypes::ElementaryPolytopTypes( const parex::Value &N_dim, const std::string &shape_names ) : dim( N_dim.ref ) {
    operations = parex::Task::call_r( "sdot/geometry/kernels/SetOfElementaryPolytops/make_ElementaryPolytopOperations(" + shape_names + ")", {} );
}

ElementaryPolytopTypes::ElementaryPolytopTypes( int dim ) : ElementaryPolytopTypes( parex::Task::ref_num( dim ), shape_names_for( dim ) ) {
}

std::string ElementaryPolytopTypes::shape_names_for( int dim ) {
    if ( dim == 3 ) return "3S 3E 4S";
    if ( dim == 2 ) return "3 4 5";
    TODO;
    return {};
}

} // namespace sdot
