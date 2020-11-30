#pragma once

#include "ElementaryPolytopInfo.h"
#include <parex/type_name.h>

/**
*/
class ElementaryPolytopInfoListContent {
public:
    using       VecElem        = std::vector<ElementaryPolytopInfo>;

    void        write_to_stream( std::ostream &os ) const { os << elem_info; }
    std::string elem_names     () const { std::string res; for( std::size_t i = 0; i < elem_info.size(); ++i ) res += ( i ? " " : "" ) + elem_info[ i ].name; return res; }

    int         default_dim;   ///<
    VecElem     elem_info;     ///<
};


inline std::string type_name( S<ElementaryPolytopInfoListContent> ) {
    return "ElementaryPolytopInfoListContent";
}
