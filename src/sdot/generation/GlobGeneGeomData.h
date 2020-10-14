#pragma once

#include <string>
#include <vector>
#include <set>

namespace sdot {

/**
*/
class GlobGeneGeomData {
public:
    using                     TI             = std::size_t;

    void                      write_gen_decls( std::string filename );
    void                      write_gen_defs ( std::string filename, bool gpu );
    std::string               mk_item_name   ( std::vector<TI> inds );

    void                      write_gen_decl ( std::ostream &os, std::vector<TI> inds, std::string prefix, std::string suffix );

    std::set<std::vector<TI>> needed_cut_ops;
};

}
