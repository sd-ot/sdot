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

SetOfElementaryPolytops::SetOfElementaryPolytops( const ElementaryPolytopInfoList &elementary_polytop_info, const Parm &types ) : elem_info( elementary_polytop_info.task ) {
    struct NewShapeMap : ComputableTask {
        NewShapeMap( std::vector<Rc<Task>> &&children, Memory *dst ) : ComputableTask( std::move( children ) ), dst( dst ) {
        }

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
            output_type = shape_map_type( type_name, epil, scalar_type, index_type, dst, dim );

            // find or create lib
            static GeneratedSymbolSet gls;
            auto *func = gls.get_symbol<void( ComputableTask *)>( [&]( SrcSet &sw ) {
                Src &src = sw.src( "get_NewShapeMap.cpp" );

                src.compilation_environment.includes << "<parex/ComputableTask.h>";
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

        Memory *dst;
    };

    Memory *dst = types.dst ? types.dst : &memory_cpu;

    shape_map = new NewShapeMap( {
        elementary_polytop_info.task,
        types.scalar_type.to_string(),
        types.index_type.to_string(),
        types.dim.conv_to<int>()
    }, dst );
}

void SetOfElementaryPolytops::write_to_stream( std::ostream &os ) const {
    os << Value( shape_map );
}

Type *SetOfElementaryPolytops::shape_map_type( const std::string &type_name, const ElementaryPolytopInfoListContent *epil, Type *scalar_type, Type *index_type, Memory *dst, int dim ) {
    return Task::type_factory().reg_cpp_type( type_name, [&]( CppType &ct ) {
        ct.compilation_environment.include_directories << SDOT_DIR "/ext/xtensor/install/include";
        ct.compilation_environment.include_directories << SDOT_DIR "/ext/xsimd/install/include";
        ct.compilation_environment.include_directories << SDOT_DIR "/src/asimd/src";
        ct.compilation_environment.include_directories << SDOT_DIR "/src";

        ct.compilation_environment.includes << "<sdot/geometry/internal/HomogeneousElementaryPolytopList.h>";
        ct.compilation_environment.includes << "<parex/type_name.h>";
        ct.compilation_environment.includes << "<functional>";

        ct.sub_types.push_back( scalar_type );
        ct.sub_types.push_back( index_type );

        //
        std::string allocator_TF = dst->allocator( ct.compilation_environment, scalar_type );
        std::string allocator_TI = dst->allocator( ct.compilation_environment, index_type );

        // def struct in preliminaries
        std::ostringstream pr;
        pr << "struct " << type_name << " {\n";
        pr << "    using AF = " << allocator_TF << ";\n";
        pr << "    using AI = " << allocator_TI << ";\n";
        pr << "    using TF = " << scalar_type->cpp_name() << ";\n";
        pr << "    using TI = " << index_type->cpp_name() << ";\n";
        pr << "    using HL = HomogeneousElementaryPolytopList<AF,AI>;\n";
        // ctor
        pr << "    \n";
        pr << "    " << type_name << "()";
        for( std::size_t i = 0; i < epil->elem_info.size(); ++i )
            pr << ( i ? ", " : " : " ) << "_" << epil->elem_info[ i ].name << "( allocator_TF, allocator_TI, " << epil->elem_info[ i ].nb_nodes << ", " << epil->elem_info[ i ].nb_faces << ", " << dim << " )";
        pr << " {}\n";

        // operator[]
        pr << "    \n";
        pr << "    HL *sub_list( const std::string &name ) {\n";
        for( const ElementaryPolytopInfo &elem : epil->elem_info )
            pr << "        if ( name == \"" << elem.name << "\" ) return &_" << elem.name << ";\n";
        pr << "        return nullptr;\n";
        pr << "    }\n";

        // for_each_shape_type
        pr << "    \n";
        pr << "    void for_each_shape_type( const std::function<void(const std::string &name)> &f ) {\n";
        for( const ElementaryPolytopInfo &elem : epil->elem_info )
            pr << "        f( \"" << elem.name << "\" );\n";
        pr << "    }\n";

        // write_to_stream
        pr << "    \n";
        pr << "    void write_to_stream( std::ostream &os ) const {\n";
        for( const ElementaryPolytopInfo &elem : epil->elem_info )
            pr << "        _" << elem.name << ".write_to_stream( os << \"" << elem.name << ":\", allocator_TF, allocator_TI, \"\\n  \" ); os << '\\n';\n";
        pr << "    }\n";

        // attributes
        pr << "    \n";
        pr << "    AF allocator_TF;\n";
        pr << "    AI allocator_TI;\n";
        pr << "    \n";
        for( const ElementaryPolytopInfo &elem : epil->elem_info )
            pr << "    HL _" << elem.name << ";\n";
        pr << "};\n";

        // type_name
        pr << "\n";
        pr << "inline std::string type_name( S<" << type_name << "> ) { return \"" << type_name << "\"; }\n";

        ct.compilation_environment.preliminaries << pr.str();
    } );
}

void SetOfElementaryPolytops::add_repeated( const Value &shape_name, const Value &count, const Value &coordinates, const Value &face_ids, const Value &beg_ids ) {
    shape_map = new CompiledIncludeTask( "sdot/geometry/internal/add_repeated.h", { shape_map, shape_name.task, count.task, coordinates.task, face_ids.task, beg_ids.task } );
}

void SetOfElementaryPolytops::display_vtk( const Value &filename ) const {
    scheduler.run( new CompiledIncludeTask( "sdot/geometry/internal/display_vtk.h", { filename.task, shape_map, elem_info }, {}, std::numeric_limits<double>::max() ) );
}

void SetOfElementaryPolytops::plane_cut( const Value &normals, const Value &scalar_products, const Value &cut_ids ) {
    P( normals );
}

} // namespace sdot
