#pragma once

#include "ElementaryPolytopCarac.h"

namespace sdot {

/**
*/
class ElementaryPolytopCaracList {
public:
    void write_to_stream( std::ostream &os ) const { os << elements; }

    std::vector<ElementaryPolytopCarac> elements;
};

} // namespace sdot
