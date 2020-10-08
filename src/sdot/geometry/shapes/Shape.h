#pragma once

#include <vector>
#include <string>
#include <set>

namespace sdot {

/**
*/
struct Shape {
    // given a cut case (defined by the set of outside points),
    struct Proc {

    };

    // different way to cut the shape, depending on which shape are available
    struct Cut {
        std::set<std::string> summary_of_needed_shapes;
        std::vector<Proc> procedure_for_each_cut_case;
    };

    std::vector<Cut> cut_procs;
    int nb_points = 0;
    int nb_faces  = 0;
};

} // namespace sdot

