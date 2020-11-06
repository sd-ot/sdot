#pragma once

#include "CutOp.h"

namespace parex {

struct CutOpWithNamesAndInds {
    using            TI                   = std::size_t;
    struct           Out                  { std::string shape_name; std::vector<TI> output_node_inds, output_face_inds; };

    /**/             CutOpWithNamesAndInds( TI dim );
    std::string      created_shapes       () const; ///< string representing the created shapes in canonical form
    std::size_t      nb_created           ( std::string shape_name ) const;


    std::vector<TI>  input_node_inds;
    std::vector<TI>  input_face_inds;
    std::vector<Out> outputs;
    CutOp            cut_op;
};

}
