#pragma once

#include <ostream>
#include <vector>
#include <map>

/**/
class OutputNodeList {
public:
    using TI = std::size_t;

    struct ByRefShape {
        void write_to_stream( std::ostream &os ) const;

        TI num_dst_ref_shape;
        std::vector<TI> perm_dst_nodes;
        std::vector<TI> perm_dst_shapes;
        std::vector<std::vector<std::pair<TI,TI>>> node_lists; // [ num_dst_shape ][ num_dst_node ]
    };

    void write_to_stream( std::ostream &os ) const;
    void write_function_call( std::ostream &os, TI num_case, std::vector<std::string> ref_shape_names, const std::map<TI,TI> &src_node_map ) const;
    void write_function_name( std::ostream &os ) const;
    void write_perm( std::ostream &os, const std::vector<TI> &perm ) const;

    void sort_with_first_shape_proposal( ByRefShape &nbr );
    void sort_with_fixed_src_node_perm();

    std::vector<TI> make_inv_perm( const std::vector<TI> &perm, const std::map<TI,TI> &src_node_map ) const;

    std::pair<TI,TI> summary( const std::pair<TI,TI> &src_nodes ) const;
    std::vector<std::pair<TI,TI>> summary( const std::vector<std::pair<TI,TI>> &src_node_list, const std::vector<TI> &perm_dst_nodes ) const;
    std::vector<std::vector<std::pair<TI,TI>>> summary( const ByRefShape &nbr ) const;
    std::vector<std::vector<std::vector<std::pair<TI,TI>>>> summary() const;

    std::vector<TI>         perm_src_nodes;
    std::vector<TI>         perm_nbrs;
    std::vector<ByRefShape> nbrs;
};

