#ifndef SDOT_GEN_ELEMENT_H
#define SDOT_GEN_ELEMENT_H

#include <ostream>
#include <vector>
#include <string>
#include <map>

/*
*/
class Element {
public:
    /**/ Element       ( std::string name );

    void write_vtk_info( std::ostream &os, std::string var_name );
    void write_cut_info( std::ostream &os, std::string var_name, std::map<std::string,Element> &available_output_elements );

    int  nb_nodes;
    int  nb_faces;
    int  nvi;
};

#include "Element.tcc"

#endif // SDOT_GEN_ELEMENT_H
