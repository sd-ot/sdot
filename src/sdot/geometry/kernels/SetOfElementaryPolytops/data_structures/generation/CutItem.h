#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <array>

/**
*/
struct CutItem {
    using           TI            = std::size_t;
    using           NN            = std::array<TI,2>;
    using           NS            = std::array<NN,2>;
    enum {          internal_face = -2 };
    enum {          cut_face      = -1 };

    bool            operator<     ( const CutItem &that ) const { return std::tie( nodes, faces, lengths ) < std::tie( that.nodes, that.faces, that.lengths ); }
    std::string     name          () const { return std::to_string( nodes.size() ); }

    std::vector<NN> nodes;        ///<
    std::vector<TI> faces;        ///< internal face, cut id or num face
    std::vector<NS> lengths;      ///< to compute the score if several possibilities
};