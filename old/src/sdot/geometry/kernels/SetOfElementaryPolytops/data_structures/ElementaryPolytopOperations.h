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
    struct             CutOp               { std::string operation_name; std::vector<OutCutOp> outputs; std::vector<TI> inp_node_corr, inp_face_corr; };
    using              MapStrVecVecTI      = std::map<std::string,std::vector<std::vector<TI>>>;
    using              VecVecCutOp         = std::vector<std::vector<CutOp>>;
    using              VecTI               = std::vector<TI>;
    using              VecPairTIVecTI      = std::vector<std::pair<TI,std::vector<TI>>>;
    using              VVVA4TI             = std::vector<std::vector<std::vector<std::array<TI,4>>>>;
    struct             CutInfo             {
        MapStrVecVecTI nb_output_elements; ///< [ output_elem_type ][ case_number ][ sub_case_number ]
        VecTI          nb_sub_cases;       ///< [ case_number ]
        VecVecCutOp    new_elems;          ///< [ case_number ][ sub case number ]
        VVVA4TI        lengths;            ///< [ case_number ]| sub_case_number ][ num_created_edge ] => array<points_index,4>
    };
    struct             Operations          {
        VecPairTIVecTI vtk_elements;       ///< [vtk_id + [node numbers]]
        CutInfo        cut_info;           ///<
        TI             nb_nodes;           ///<
        TI             nb_faces;           ///<
    };
    using              MapStrOp            = std::map<std::string,Operations>;

    void               write_to_stream     ( std::ostream &os ) const { for( const auto &p : operation_map ) os << p.first << " "; }
    static std::string type_name           () { return "sdot::ElementaryPolytopOperations"; }

    MapStrOp           operation_map;      ///< inp elem => operations
};

} // namespace sdot
