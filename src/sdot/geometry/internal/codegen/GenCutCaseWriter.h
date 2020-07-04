#pragma once

#include <vector>
#include <array>
#include <map>

/***/
class GenCutCaseWriter {
public:
    using                      TI              = std::size_t;
    struct                     Output          { std::vector<std::array<TI,2>> inds; };
    struct                     ByOutputShape   { std::string shape_name; std::vector<Output> outputs; };

    /**/                       GenCutCaseWriter();

    ByOutputShape&             by_output_shape ( std::string shape_name );
    void                       write_func_name ( std::ostream &os );
    void                       write_func_args ( std::ostream &os, TI num_case );
    void                       write_to        ( std::ostream &os, TI num_case );
    bool                       operator<       ( const GenCutCaseWriter &that ) const;
    void                       optimize        ();

    void                       optimize_src_map();

    std::vector<ByOutputShape> by_output_shapes;
    std::vector<TI>            num_src_nodes;
    std::map<TI,TI>            src_map;
};

