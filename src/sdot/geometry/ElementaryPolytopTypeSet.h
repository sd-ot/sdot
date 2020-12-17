#pragma once

#include <parex/Vector.h>

namespace sdot {

/**
*/
class ElementaryPolytopTypeSet {
public:
    /**/                       ElementaryPolytopTypeSet( const parex::String &list_of_shape_tpes );
    /**/                       ElementaryPolytopTypeSet( const parex::Scalar &dim );

    void                       write_to_stream         ( std::ostream &os ) const;

    parex::Rc<parex::Variable> variable;
};

} // namespace sdot
