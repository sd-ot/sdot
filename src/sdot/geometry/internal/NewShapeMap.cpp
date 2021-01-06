#include <parex/instructions/NotOwnedSrcInstruction.h>
#include <parex/compilation/variable_encode.h>
#include <parex/data/CompiledType.h>
#include <parex/data/TypeFactory.h>
#include <parex/utility/TODO.h>

#include "ElementaryPolytopCaracList.h"
#include "NewShapeMap.h"

namespace sdot {

NewShapeMap::NewShapeMap( const ElementaryPolytopTypeSet &elementary_polytop_type_set, const parex::String &scalar_type, const parex::String &index_type, const parex::Scalar &dim, parex::Memory *dst ) :
    parex::CompiledInstruction( "NewShapeMap", {
        elementary_polytop_type_set.carac->get(),
        scalar_type.to_string().expr(),
        index_type.to_string().expr(),
        dim.to<int>().expr(),
        parex::Expression{ new parex::NotOwnedSrcInstruction( dst->name(), dst ), 0 }
    }, 1 ), dst( dst ) {
}

void NewShapeMap::get_src_content( parex::Src &src, parex::SrcSet &, parex::TypeFactory *tf ) const {
    parex::Type *sm = tf->type( output_type_name() );
    src.compilation_environment += sm->compilation_environment;

    src << "template<class Carac>\n";
    src << sm->name << " *" << called_func_name() << "( const Carac &, const std::string &, const std::string &, const int &, " << dst->name() << " &memory ) {\n";
    src << "    return new " << sm->name << "( &memory, &memory, /*rese*/ 0 );\n";
    src << "}\n";
}

void NewShapeMap::prepare( parex::TypeFactory *tf, parex::SchedulerSession * ) {
    // create the output type if necessary
    tf->reg_type( output_type_name(), [&]( const std::string &type_name ) {
        auto& epcl        = *reinterpret_cast<ElementaryPolytopCaracList *>( slots[ 0 ].input.data_ptr() );
        auto& scalar_type = *reinterpret_cast<std::string *>( slots[ 1 ].input.data_ptr() );
        auto& index_type  = *reinterpret_cast<std::string *>( slots[ 2 ].input.data_ptr() );
        auto  dim         = *reinterpret_cast<int *>( slots[ 3 ].input.data_ptr() );
        auto  al_TF       = dst->allocator_type( scalar_type, index_type );
        auto  al_TI       = dst->allocator_type( index_type, index_type );

        if ( dim == 0 )
            dim = epcl.elements[ 0 ].nvi;

        std::ostringstream decl;
        decl << "struct " << type_name << " {\n";
        // using ...
        decl << "    using TF = " << scalar_type << ";\n";
        decl << "    using TI = " << index_type << ";\n";
        decl << "    using AF = " << al_TF << ";\n";
        decl << "    using AI = " << al_TI << ";\n";

        // using ... = homogeneous list
        decl << "\n";
        for( const ElementaryPolytopCarac &element : epcl.elements ) {
            decl << "    using TL_" << element.name << " = HomogeneousElementaryPolytopList<AF,AI,"
                 << element.nb_nodes << ","
                 << element.nb_faces << ","
                 << element.nvi
                 << ">;\n";
        }

        // ctor
        decl << "\n";
        decl << "    " << type_name << "( const AF &allocator_TF, const AI &allocator_TI, TI rese_items = 0 )";
        for( std::size_t i = 0; i < epcl.elements.size(); ++i )
            decl << ( i ? ", _" : " : _" ) << epcl.elements[ i ].name << "( allocator_TF, allocator_TI, rese_items )";
        decl << " {}\n";

        // apply
        decl << "\n";
        decl << "    template<class Func> auto apply_on( std::string shape_name, Func &&func ) {\n";
        for( std::size_t i = 1; i < epcl.elements.size(); ++i )
            decl << "if ( shape_name == \"" << epcl.elements[ i ].name << "\" ) return func( _" << epcl.elements[ i ].name << " );\n";
        decl << "        ASSERT( shape_name == \"" << epcl.elements[ 0 ].name << "\", \"no element named {}\", shape_name );\n";
        if ( epcl.elements.size() )
            decl << "        return func( _" << epcl.elements[ 0 ].name << " );\n";
        decl << "    }\n";

        //        // for_each_shape_type
        //        pr << "    \n";
        //        pr << "    void for_each_shape_type( const std::function<void(const std::string &name)> &f ) {\n";
        //        for( const ElementaryPolytopInfo &elem : epil->elem_info )
        //            pr << "        f( \"" << elem.name << "\" );\n";
        //        pr << "    }\n";

        // write_to_stream
        decl << "    \n";
        decl << "    void write_to_stream( std::ostream &os ) const {\n";
        for( std::size_t i = 0; i < epcl.elements.size(); ++i )
            decl << "        _" << epcl.elements[ i ].name << ".write_to_stream( os << \"" << epcl.elements[ i ].name << ":\", \"\\n  \" ); os << '\\n';\n";
        decl << "    }\n";

        // attributes
        decl << "\n";
        for( const ElementaryPolytopCarac &element : epcl.elements )
            decl << "    TL_" << element.name << " _" << element.name << ";\n";
        decl << "};\n";
        decl << "PAREX_DECL_HOMO_TYPE_INFO( " << type_name << " );\n";

        parex::Type *res = new parex::CompiledType( type_name, {}, {}, /*sub types*/ {} );
        res->compilation_environment.include_directories << SDOT_DIR "/src";
        res->compilation_environment.includes << "<sdot/geometry/internal/HomogeneousElementaryPolytopList.h>";
        res->compilation_environment.includes << "<parex/utility/ASSERT.h>";
        res->compilation_environment.includes << "<parex/data/TypeInfo.h>";
        res->compilation_environment.preliminaries << decl.str();
        return res;
    } );
}

std::string NewShapeMap::summary() const {
    return "NewShapeMap " + output_type_name();
}

std::string NewShapeMap::output_type_name() const {
    auto ct          = slots[ 0 ].input.first_slot()->data->type;
    auto scalar_type = *reinterpret_cast<std::string *>( slots[ 1 ].input.data_ptr() );
    auto index_type  = *reinterpret_cast<std::string *>( slots[ 2 ].input.data_ptr() );
    auto dim         = *reinterpret_cast<int *>( slots[ 3 ].input.data_ptr() );

    std::string res = "ShapeMap";
    res += "_" + parex::variable_encode( dst->name(), true );
    res += "_" + parex::variable_encode( scalar_type, true );
    res += "_" + parex::variable_encode( index_type , true );
    res += "_" + parex::variable_encode( ct->name   , true );
    res += "_" + std::to_string( dim );
    return res;
}

} // namespace sdot
