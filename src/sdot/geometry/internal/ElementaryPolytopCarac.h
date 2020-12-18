#pragma once

#include <parex/utility/generic_ostream_output.h>
#include <vector>

namespace sdot {

/**
*/
class ElementaryPolytopCarac {
public:
    using       VtkElements    = std::vector<std::pair<unsigned,std::vector<unsigned>>>; ///< [ vtk_type => list of nodes ]

    void        write_to_stream( std::ostream &os ) const { os << name << "(nb_nodes=" << nb_nodes << ")"; }

    VtkElements vtk_elements;  ///<
    unsigned    nb_nodes;      ///<
    unsigned    nb_faces;      ///<
    std::string name;          ///<
    unsigned    nvi;           ///< nb var inter
};

} // namespace sdot
