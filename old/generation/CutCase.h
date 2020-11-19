#pragma once

#include "NamedRecursivePolytop.h"

namespace sdot {

struct CutItem {
    using           TI            = std::size_t;
    using           NN            = std::array<TI,2>;
    using           NS            = std::array<NN,2>;
    enum {          internal_face = -2 };
    enum {          cut_face      = -1 };

    bool            operator<     ( const CutItem &that ) const { return std::tie( nodes, faces, lengths ) < std::tie( that.nodes, that.faces, that.lengths ); }

    std::vector<NN> nodes;        ///<
    std::vector<TI> faces;        ///< internal face, cut id or num face
    std::vector<NS> lengths;      ///< to compute the score if several possibilities
};

struct CutOp {
    std::string          mk_item_func_name() const;
    std::size_t          nb_output_shapes () const { return cut_items.size(); }
    std::size_t          nb_input_nodes   () const;
    std::size_t          nb_input_faces   () const;
    bool                 operator<        ( const CutOp &that ) const;
    operator             bool             () const;

    std::vector<CutItem> cut_items;
    std::size_t          dim;
};

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

/**
*/
class CutCase {
public:
    using             TI            = std::size_t;
    struct            IndOut        { TI ind_0, ind_1, face_id; bool outside; bool operator<( const IndOut &that ) const; void write_to_stream( std::ostream &os ) const; bool plain() const; };
    using             Upp           = std::unique_ptr<CutOpWithNamesAndInds>;

    void              init          ( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &primitive_shapes );

    void              _init_2D      ( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &primitive_shapes );
    void              _init_2D_rec  ( CutOpWithNamesAndInds &possibility, const std::vector<IndOut> &points, const std::vector<NamedRecursivePolytop> &primitive_shapes );
    bool              _has_2D_shape ( TI nb_points, const std::vector<NamedRecursivePolytop> &primitive_shapes );

    std::vector<Upp>  possibilities;
    std::vector<bool> out_points;
};

}
