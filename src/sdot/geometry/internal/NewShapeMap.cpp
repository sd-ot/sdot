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
        dim.to<int>().expr()
    }, 1 ), dst( dst ) {
}

void NewShapeMap::prepare( parex::TypeFactory *tf, parex::SchedulerSession * ) {
    // create the output type if necessary
    tf->reg_type( output_type_name(), [&]( const std::string &type_name ) {
        auto epcl = reinterpret_cast<ElementaryPolytopCaracList *>( slots[ 0 ].input.first_slot()->data->ptr );

        std::ostringstream decl;
        decl << "struct " << type_name << " {\n";
        for( const ElementaryPolytopCarac &element : epcl->elements )
        decl << "    HomogeneousElementaryPolytopList<> _" << element.name << ";\n";
        decl << "};\n";
        decl << "PAREX_DECL_HOMO_TYPE_INFO( " << type_name << " );\n";

        parex::Type *res = new parex::CompiledType( type_name, {}, {}, /*sub types*/ {} );
        res->compilation_environment.includes << "<sdot/geometry/internal/HomogeneousElementaryPolytopList.h>";
        res->compilation_environment.includes << "<parex/data/TypeInfo.h>";
        res->compilation_environment.preliminaries << decl.str();
        return res;
    } );
}

std::string NewShapeMap::output_type_name() const {
    parex::Type *ct = slots[ 0 ].input.first_slot()->data->type;
    return "ShapeMap_" + ct->name;
}

void NewShapeMap::get_src_content( parex::Src &src, parex::SrcSet &, parex::TypeFactory *tf ) const {
    src.compilation_environment += tf->type( output_type_name() )->compilation_environment;
}

//        virtual void exec() override {
//            // inputs
//            const ElementaryPolytopInfoListContent *epil = reinterpret_cast<const ElementaryPolytopInfoListContent *>( children[ 0 ]->output_data );
//            Type *scalar_type = Task::type_factory( *reinterpret_cast<const std::string *>( children[ 1 ]->output_data ) );
//            Type *index_type = Task::type_factory( *reinterpret_cast<const std::string *>( children[ 2 ]->output_data ) );
//            int dim = *reinterpret_cast<const int *>( children[ 3 ]->output_data );
//            if ( ! dim )
//                dim = epil->default_dim;

//            // type name
//            std::string type_name = "ShapeMap";
//            type_name += "_" + variable_encode( epil->elem_names(), true );
//            type_name += "_" + variable_encode( scalar_type->cpp_name(), true );
//            type_name += "_" + variable_encode( index_type->cpp_name(), true );
//            type_name += "_" + std::to_string( dim );

//            // set output type
//            output_type = shape_map_type( type_name, epil, scalar_type, index_type, dst, dim );

//            // find or create lib
//            static GeneratedSymbolSet gls;
//            auto *func = gls.get_symbol<void( ComputableTask *)>( [&]( SrcSet &sw ) {
//                Src &src = sw.src( "get_NewShapeMap.cpp" );

//                src.compilation_environment.includes << "<parex/ComputableTask.h>";
//                output_type->add_needs_in( src );

//                src << "\n";
//                src << "extern \"C\" void exported( ComputableTask *task ) {\n";
//                src << "    task->output.data = new " << type_name << ";\n";
//                src << "    task->output.own = true;\n";
//                src << "}\n";
//            }, type_name );

//            // execute the generated function to get the output_data
//            func( this );
//        }

} // namespace sdot
