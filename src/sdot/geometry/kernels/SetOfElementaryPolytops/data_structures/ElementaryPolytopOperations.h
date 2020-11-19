#pragma once

#include <ostream>
#include <vector>
#include <string>
#include <map>

namespace sdot {


/**
*/
class ElementaryPolytopOperations {
public:
    using              TI                  = std::size_t;
    struct             OutCutOp            { std::string shape_type; std::vector<TI> node_corr, face_corr; };
    struct             CutOp               { std::string operation_name; std::vector<OutCutOp> outputs; std::vector<TI> inp_node_corr, inp_face_corr; TI num_case, num_sub_case; };
    using              MapStrVecTI         = std::map<std::string,std::vector<TI>>;
    using              VecCutOp            = std::vector<CutOp>;
    using              VecTI               = std::vector<TI>;
    using              VecPairTIVecTI      = std::vector<std::pair<TI,std::vector<TI>>>;
    struct             CutInfo             {
        MapStrVecTI    nb_output_elements; ///< [output_elem_type][case_and_sub_case_number] => nb output elements
        VecTI          nb_sub_cases;       ///< [case_number] => nb_sub_cases
        VecCutOp       new_elems;          ///< new elements for each [case + sub case number]
    };
    struct             Operations          {
        VecPairTIVecTI vtk_elements;                  ///< [vtk_id + [node numbers]]
        CutInfo        cut_info;
        TI             nb_nodes;
        TI             nb_faces;
    };
    using              MapStrOp            = std::map<std::string,Operations>;

    void               write_to_stream     ( std::ostream &os ) const { for( const auto &p : operation_map ) os << p.first << " "; }
    static std::string type_name           () { return "sdot::ElementaryPolytopOperations"; }

    MapStrOp           operation_map;
};

} // namespace sdot
