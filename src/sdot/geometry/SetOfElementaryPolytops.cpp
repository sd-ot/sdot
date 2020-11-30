#include <parex/CompiledIncludeTask.h>
#include <parex/GeneratedSymbolSet.h>
#include <parex/variable_encode.h>
#include <parex/Scheduler.h>
#include <parex/CppType.h>
#include <parex/TODO.h>
#include <parex/P.h>
#include <sstream>

#include "internal/ElementaryPolytopInfoListContent.h"
#include "SetOfElementaryPolytops.h"

namespace sdot {

SetOfElementaryPolytops::SetOfElementaryPolytops( const ElementaryPolytopInfoList &elementary_polytop_info, const Value &scalar_type, const Value &index_type, const Value &dim ) {
    struct NewShapeMap : ComputableTask {
        using ComputableTask::ComputableTask;

        virtual void write_to_stream( std::ostream &os ) const override {
            os << "GetShapeMap";
        }

        virtual void exec() override {
            // inputs
            const ElementaryPolytopInfoListContent *epil = reinterpret_cast<const ElementaryPolytopInfoListContent *>( children[ 0 ]->output_data );
            Type *scalar_type = Task::type_factory( *reinterpret_cast<const std::string *>( children[ 1 ]->output_data ) );
            Type *index_type = Task::type_factory( *reinterpret_cast<const std::string *>( children[ 2 ]->output_data ) );
            int dim = *reinterpret_cast<const int *>( children[ 3 ]->output_data );
            if ( ! dim )
                dim = epil->default_dim;

            // type name
            std::string type_name = "ShapeMap";
            type_name += "_" + variable_encode( epil->elem_names(), true );
            type_name += "_" + variable_encode( scalar_type->cpp_name(), true );
            type_name += "_" + variable_encode( index_type->cpp_name(), true );
            type_name += "_" + std::to_string( dim );

            // set output type
            output_type = shape_map_type( type_name, epil, scalar_type, index_type, dim );

            // find or create lib
            static GeneratedSymbolSet gls;
            auto *func = gls.get_symbol<void( ComputableTask *)>( [&]( SrcSet &sw ) {
                Src &src = sw.src( "get_NewShapeMap.cpp" );
                src.includes << "<parex/ComputableTask.h>";
                output_type->add_needs_in( src );

                src << "\n";
                src << "extern \"C\" void exported( ComputableTask *task ) {\n";
                src << "    task->output_data = new " << type_name << ";\n";
                src << "    task->output_own = true;\n";
                src << "}\n";
            }, type_name );

            // execute the generated function to get the output_data
            func( this );
        }
    };

    shape_map = new NewShapeMap( {
        elementary_polytop_info.task,
        scalar_type.to_string(),
        index_type.to_string(),
        dim.conv_to<int>()
    } );
}

void SetOfElementaryPolytops::write_to_stream( std::ostream &os ) const {
    os << Value( shape_map );
}

Type *SetOfElementaryPolytops::shape_map_type( const std::string &type_name, const ElementaryPolytopInfoListContent *epil, Type *scalar_type, Type *index_type, int dim ) {
    return Task::type_factory().reg_cpp_type( type_name, [&]( CppType &ct ) {
        ct.includes << "<sdot/geometry/internal/HomogeneousElementaryPolytopList.h>";
        ct.include_directories << SDOT_DIR "/ext/xtensor/install/include";
        ct.include_directories << SDOT_DIR "/ext/xsimd/install/include";
        ct.include_directories << SDOT_DIR "/src";

        ct.sub_types.push_back( scalar_type );
        ct.sub_types.push_back( index_type );

        // preliminaries
        std::ostringstream pr;
        pr << "struct " << type_name << " {\n";
        pr << "    using TF = " << scalar_type->cpp_name() << ";\n";
        pr << "    using TI = " << index_type->cpp_name() << ";\n";
        pr << "    using HL = HomogeneousElementaryPolytopList<TF,TI>;\n";
        pr << "    \n";
        pr << "    HL *operator[]( const std::string &name ) const {\n";
        for( const ElementaryPolytopInfo &elem : epil->elem_info )
            pr << "        if ( name == \"" << elem.name << "\" ) return &_" << elem.name << ";\n";
        pr << "        return nullptr;\n";
        pr << "    }\n";
        pr << "    \n";
        pr << "    void write_to_stream( std::ostream &os ) const {\n";
        for( const ElementaryPolytopInfo &elem : epil->elem_info )
            pr << "        _" << elem.name << ".write_to_stream( os << \"" << elem.name << ":\", \"\\n  \" ); os << '\\n';\n";
        pr << "    }\n";
        pr << "    \n";
        for( const ElementaryPolytopInfo &elem : epil->elem_info )
            pr << "    HL _" << elem.name << ";\n";
        pr << "};\n";
        ct.preliminaries << pr.str();
    } );
}

void SetOfElementaryPolytops::add_repeated( const Value &shape_name, const Value &count, const Value &coordinates, const Value &face_ids, const Value &beg_ids ) {
    shape_map = new CompiledIncludeTask( "sdot/geometry/internal/add_repeated.h", { shape_map, shape_name.task, count.task, coordinates.task, face_ids.task, beg_ids.task } );
}

} // namespace sdot
