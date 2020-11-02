#pragma once

#include "CutOp.h"
#include <set>

namespace sdot {

/**
*/
class GlobGeneGeomData {
public:
    using           TI                   = std::size_t;
    struct          Usd                  { std::vector<std::array<std::array<std::size_t,2>,2>> points; TI dim; bool operator<( const Usd &that ) const { return std::tie( points, dim ) < std::tie( that.points, that.dim ); } };

    void            write_gen_decls      ( std::string filename );
    void            write_gen_defs       ( std::string filename, bool gpu );

    void            write_gen_decl       ( std::ostream &os, const CutOp &cut_op, std::string prefix, std::string suffix );
    void            write_gen_def_mk_item( std::ostream &os, bool gpu, const CutOp &cut_op );

    std::set<CutOp> needed_cut_ops;
    std::set<int>   possible_dims;
};

}
