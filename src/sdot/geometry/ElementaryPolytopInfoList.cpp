#include "internal/GetElementaryPolytopInfoList.h"
#include "ElementaryPolytopInfoList.h"

namespace sdot {

ElementaryPolytopInfoList::ElementaryPolytopInfoList( const Value &dim_or_shape_types ) {
    task = new GetElementaryPolytopInfoList( { dim_or_shape_types.to_string() } );
}

void ElementaryPolytopInfoList::write_to_stream( std::ostream &os ) const {
    os << Value( task );
}

} // namespace sdot
