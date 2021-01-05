#include "GetElementaryPolytopCaracList.h"
#include "SymbolicElementaryPolytop.h"

#include <parex/utility/va_string.h>
#include <parex/utility/ASSERT.h>

#include <parex/data/CompiledType.h>
#include <parex/data/TypeFactory.h>

namespace sdot {

GetElementaryPolytopCaracList::GetElementaryPolytopCaracList( const parex::Vector<parex::String> &shape_names ) :
    parex::CompiledInstruction( "GetElementaryPolytopCaracList", { shape_names.to<std::vector<std::string>>().variable->get() }, 1 ) {
}

void GetElementaryPolytopCaracList::prepare( parex::TypeFactory *tf, parex::SchedulerSession *) {
    // create type if necessary
    tf->reg_type( type_name(), [&]( const std::string & ) {
        std::ostringstream decl;
        decl << "struct " << type_name() << " : sdot::ElementaryPolytopCaracList {};\n";
        decl << "PAREX_DECL_HOMO_TYPE_INFO( " << type_name() << " );\n";

        parex::Type *res = new parex::CompiledType( type_name(), {}, {}, /*sub types*/ {} );
        res->compilation_environment.includes << "<sdot/geometry/internal/ElementaryPolytopCaracList.h>";
        res->compilation_environment.includes << "<parex/data/TypeInfo.h>";

        res->compilation_environment.include_directories << SDOT_DIR "/src";

        res->compilation_environment.preliminaries << decl.str();
        return res;
    } );
}

void GetElementaryPolytopCaracList::get_src_content( parex::Src &src, parex::SrcSet &, parex::TypeFactory *tf ) const {
    // type decl
    parex::Type *type = tf->type( type_name() );
    src.compilation_environment += type->compilation_environment;

    // symbolic elements, constructed from names
    std::vector<SymbolicElementaryPolytop> ses;
    for( const std::string &shape_name : shape_names() )
        ses.push_back( shape_name );

    // maker
    src << "    " << type->name << " make_" << type->name << "() {\n";
    src << "        " << type->name << " res;\n";
    for( const SymbolicElementaryPolytop &se : ses )
        write_carac( src << "\n", se );
    src << "        \n";
    src << "        return res;\n";
    src << "    }\n";

    //
    src << "template<class T>\n";
    src << "auto kernel( const T &shape_names ) {\n";
    src << "    static " << type->name << " res = make_" << type->name << "();\n";
    src << "    return parex::MakeOutputSlots::NotOwned<" << type->name << ">{ &res };\n";
    src << "}\n";
}

std::string GetElementaryPolytopCaracList::summary() const {
    return parex::va_string( "GetElementaryPolytopCaracList {}", shape_names() );
}

const std::vector<std::string> &GetElementaryPolytopCaracList::shape_names() const {
    ASSERT( slots[ 0 ].input.first_slot() && slots[ 0 ].input.first_slot()->data, "" );
    return *reinterpret_cast<std::vector<std::string> *>( slots[ 0 ].input.data_ptr() );
}

std::string GetElementaryPolytopCaracList::type_name() const {
    std::string res = "ElementaryPolytopCaracList";
    for( const std::string &shape_name : shape_names() )
        res += "_" + shape_name;
    return res;
}

void GetElementaryPolytopCaracList::write_carac( parex::Src &src, const SymbolicElementaryPolytop &se ) const {
    std::string vn = "ep_" + se.name;

    src << "        sdot::ElementaryPolytopCarac " + vn + ";\n";
    src << "        " + vn + ".vtk_elements = " << se.vtk_output() << ";\n";
    src << "        " + vn + ".nb_nodes = " << se.nb_nodes() << ";\n";
    src << "        " + vn + ".nb_faces = " << se.nb_faces() << ";\n";
    src << "        " + vn + ".name = \"" << se.name << "\";\n";
    src << "        " + vn + ".nvi = " << se.nvi << ";\n";
    src << "        res.elements.push_back( std::move( " + vn + " ) );\n";
}


} // namespace sdot
