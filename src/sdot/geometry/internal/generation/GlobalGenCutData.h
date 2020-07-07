#pragma once

#include <set>
#include <map>

/**/
struct GlobalGenCutData {
    using TI = std::size_t;

    struct PartInfo {
        TI           nb_faces;
        TI           nb_nodes;
        std::set<TI> dims;
    };

    std::map<std::string,PartInfo>     parts;
    std::map<std::string,std::set<TI>> needed_vec_ops_function_names; // name => dims
};
