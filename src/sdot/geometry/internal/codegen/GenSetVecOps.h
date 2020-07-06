#pragma once

#include <vector>
#include <string>
#include <array>

/**
*/
class GenSetVecOps {
public:
    using  TI                  = std::size_t;

    /**/   GenSetVecOps        ( std::string func_name, TI dim );

    void   write               ( std::ostream &os ) const;

private:
    using  Node                = std::array<TI,2>;

    struct Output              {
        std::vector<Node>      nodes;
    };

    struct ByOutputShape       {
        std::vector<Output>    outputs;
    };

    std::set<TI>               needed_src_inds;
    TI                         nb_src_nodes;
    std::string                func_name;
    TI                         dim;
    std::vector<ByOutputShape> nl;
};
