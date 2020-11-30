#pragma once

#include <parex/generic_ostream_output.h>
#include <vector>

/**
*/
class ElementaryPolytopInfo {
public:
    using       VtkElements    = std::vector<std::pair<unsigned,std::vector<unsigned>>>; ///< [ vtk_type => list of nodes ]

    void        write_to_stream( std::ostream &os ) const { os << name; }

    VtkElements vtk_elements;  ///<
    unsigned    nb_nodes;      ///<
    unsigned    nb_faces;      ///<
    std::string name;          ///<
};

