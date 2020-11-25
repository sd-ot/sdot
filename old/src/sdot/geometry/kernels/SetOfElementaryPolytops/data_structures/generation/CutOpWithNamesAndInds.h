#ifndef SDOT_CutOpWithNamesAndInds_H
#define SDOT_CutOpWithNamesAndInds_H

#include <functional>
#include "CutOp.h"

/**
*/
struct CutOpWithNamesAndInds {
    using            TI                   = std::size_t;
    struct           Out                  { std::string shape_name; std::vector<TI> output_node_inds, output_face_inds; };

    /**/             CutOpWithNamesAndInds( TI dim );
    void             for_each_new_edge    ( std::function<void( TI n00, TI n01, TI n10, TI n11 )> f ) const;
    std::string      created_shapes       () const; ///< string representing the created shapes in canonical form
    std::size_t      nb_created           ( std::string shape_name ) const;


    std::vector<TI>  input_node_inds;
    std::vector<TI>  input_face_inds;
    std::vector<Out> outputs;
    CutOp            cut_op;
};

#include "CutOpWithNamesAndInds.tcc"

#endif // SDOT_CutOpWithNamesAndInds_H
