#pragma once

#include <parex/Value.h>

namespace sdot {

/**
*/
class ElementaryPolytopInfoList {
public:
    /**/     ElementaryPolytopInfoList( const Value &dim_or_shape_types );

    void     write_to_stream          ( std::ostream &os ) const;

    Rc<Task> task;
};

} // namespace sdot
