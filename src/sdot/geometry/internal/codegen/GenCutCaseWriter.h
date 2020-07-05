#pragma once

#include <vector>
#include <array>
#include <map>

/***/
class GenCutCaseWriter {
public:
    using                      TI              = std::size_t;
    struct                     Output          { std::vector<std::array<TI,2>> inds; };
    struct                     ByOutputShape   { std::string shape_name; std::vector<Output> outputs; std::vector<TI> num_dst_vertex; };

    /**/                       GenCutCaseWriter();

    ByOutputShape&             by_output_shape ( std::string shape_name, TI nb_vertices );
    void                       write_func_name ( std::ostream &os, const ByOutputShape &bos, TI n ) const;
    void                       write_func_name ( std::ostream &os ) const;
    void                       write_func_args ( std::ostream &os, TI num_case ) const;
    bool                       operator<       ( const GenCutCaseWriter &that ) const;
    void                       write_to        ( std::ostream &os, TI num_case );
    void                       optimize        ();

    void                       optimize_src_map();
    void                       optimize        ( ByOutputShape &bos );

    std::vector<ByOutputShape> by_output_shapes;
    std::map<TI,TI>            src_map;
};

