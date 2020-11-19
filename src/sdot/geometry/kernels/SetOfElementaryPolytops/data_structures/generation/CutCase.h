#ifndef SDOT_CutCase_H
#define SDOT_CutCase_H

#include "CutOpWithNamesAndInds.h"
#include "Element.h"

/**
*/
struct CutCase {
    using             TI            = std::size_t;
    struct            IndOut        { TI ind_0, ind_1, face_id; bool outside; bool operator<( const IndOut &that ) const; void write_to_stream( std::ostream &os ) const; bool plain() const; };
    using             Upp           = std::unique_ptr<CutOpWithNamesAndInds>;

    void              init          ( const Element &rp, const std::vector<bool> &out_points, std::map<std::string,Element> &primitive_shapes );

    void              _init_2D      ( const Element &rp, const std::vector<bool> &out_points, std::map<std::string,Element> &primitive_shapes );
    void              _init_2D_rec  ( CutOpWithNamesAndInds &possibility, const std::vector<IndOut> &points, std::map<std::string,Element> &primitive_shapes );
    bool              _has_2D_shape ( int nb_points, std::map<std::string,Element> &primitive_shapes );

    std::vector<Upp>  possibilities;
    std::vector<bool> out_points;
};

#include "CutCase.tcc"

#endif // SDOT_CutCase_H
