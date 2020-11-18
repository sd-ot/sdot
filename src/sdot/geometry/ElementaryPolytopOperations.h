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
    using              TI             = std::size_t;

    struct             OutCutOp       { std::string shape_type; std::vector<TI> node_corr, face_corr; };
    struct             CutOp          { std::string operation_name; std::vector<OutCutOp> outputs; std::vector<TI> inp_node_corr, inp_face_corr; TI num_case, num_sub_case; };

    void               write_to_stream( std::ostream &os ) const { for( const auto &p : operation_map ) os << p.first << " "; }

    static std::string type_name      () { return "sdot::ElementaryPolytopOperations"; }

    struct Operations {
        std::vector<CutOp>                         new_element_creation_cut; ///< creation of new elements
        std::map<std::string,std::vector<TI>>      new_output_elements_cut;  ///< [output_elem_type][case_and_sub_case_number] => nb output elements
        std::vector<TI>                            nb_sub_cases_cut;         ///< [case_number] => nb_sub_cases
        std::vector<std::pair<TI,std::vector<TI>>> vtk_elements;             ///< [vtk_id + [node numbers]]
        TI                                         nb_nodes;
        TI                                         nb_faces;
    };

    std::map<std::string,Operations> operation_map;
};

} // namespace sdot
