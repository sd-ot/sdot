#pragma once

#include "CutOp.h"
#include <set>

namespace sdot {

/**
*/
class GlobGeneGeomData {
public:
    using           TI             = std::size_t;

    void            write_gen_decls( std::string filename );
    void            write_gen_defs ( std::string filename, bool gpu );

    void            write_gen_decl ( std::ostream &os, const CutOp &cut_op, std::string prefix, std::string suffix );

    std::set<CutOp> needed_cut_ops;
};

}
