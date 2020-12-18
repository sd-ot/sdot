#pragma once

#include <parex/Vector.h>

namespace sdot {

/**
*/
class ElementaryPolytopTypeSet {
public:
    using                      VS                      = parex::Vector<parex::String>;
    using                      SC                      = parex::Scalar;

    /**/                       ElementaryPolytopTypeSet( const VS &shape_names );
    /**/                       ElementaryPolytopTypeSet( const SC &dim );

    static VS                  default_shape_names_for ( const SC &dim );
    void                       write_to_stream         ( std::ostream &os ) const;

    parex::Rc<parex::Variable> carac;        ///<
};

} // namespace sdot
