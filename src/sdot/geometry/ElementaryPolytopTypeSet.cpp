#include <parex/instructions/CompiledLambdaInstruction.h>
#include <parex/compilation/GeneratedSymbolSet.h>
#include <parex/utility/va_string.h>
#include <parex/utility/ASSERT.h>
#include "ElementaryPolytopTypeSet.h"

#include <parex/utility/P.h>

namespace sdot {

ElementaryPolytopTypeSet::ElementaryPolytopTypeSet( const parex::Vector<parex::String> &shape_names ) {
    struct GetElementaryPolytopCaracList : parex::CompiledInstruction {
    public:
        using parex::CompiledInstruction::CompiledInstruction;

        virtual void get_src_content( parex::Src &src, parex::SrcSet &/*sw*/ ) const override {
            // Pb 1: il faudrait générer le code en fonction du contenu. Prop: on passe par summary
        }

        virtual std::string summary() const override {
            return parex::va_string( "GetElementaryPolytopCaracList {}", shape_names() );
        }

        std::vector<std::string> &shape_names() const {
            ASSERT( slots[ 0 ].input.first_slot() && slots[ 0 ].input.first_slot()->data, "" );
            return *reinterpret_cast<std::vector<std::string> *>( slots[ 0 ].input.data_ptr() );
        }
    };
    variable = new parex::Variable( new GetElementaryPolytopCaracList( "GetElementaryPolytopCaracList", { shape_names.to<std::vector<std::string>>().variable->get() }, 1 ), 1 );

    //        virtual void exec() override {
    //            // set the output type
    //            output_type = type_factory().reg_cpp_type( "ElementaryPolytopInfoListContent", []( CppType &ct ) {
    //                ct.compilation_environment.includes << "<sdot/geometry/internal/ElementaryPolytopInfoListContent.h>";
    //                ct.compilation_environment.include_directories << SDOT_DIR "/src";
    //            } );

    //            // summary
    //            std::string shape_types = *reinterpret_cast<const std::string *>( children[ 0 ]->output_data );

    //            // find or create lib
    //            static GeneratedSymbolSet gls;
    //            auto *func = gls.get_symbol<void( ComputableTask *)>( [&]( SrcSet &sw ) {
    //                Src &src = sw.src( "get_ElementaryPolytopInfoList.cpp" );

    //                // src.cpp_flags << "-std=c++17" << "-g3";
    //                src.compilation_environment.includes << "<parex/ComputableTask.h>";
    //                output_type->add_needs_in( src );

    //                src << "namespace {\n";
    //                src << "struct Epil : ElementaryPolytopInfoListContent {\n";
    //                src << "    Epil() {";
    //                write_epil_ctor( src, shape_types, "\n        " );
    //                src << "    }\n";
    //                src << "};\n";
    //                src << "}\n";

    //                src << "\n";
    //                src << "extern \"C\" void exported( ComputableTask *task ) {\n";
    //                src << "    static Epil res;\n";
    //                src << "    task->output.data = &res;\n";
    //                src << "    task->output.own = false;\n";
    //                src << "}\n";
    //            }, shape_types );

    //            // execute the generated function to get the output_data
    //            func( this );
    //        }

    //        void write_epil_ctor( Src &src, const std::string &shape_types, const std::string &sp ) {
    //            std::vector<SymbolicElementaryPolytop> lst;
    //            std::istringstream ss( shape_types );
    //            for( std::string name; ss >> name; )
    //                lst.push_back( name );

    //            src << sp << "default_dim = 2;";

    //            for( std::size_t num_elem = 0; num_elem < lst.size(); ++num_elem ) {
    //                SymbolicElementaryPolytop &se = lst[ num_elem ];
    //                src << sp << "";
    //                src << sp << "ElementaryPolytopInfo e_" << num_elem << ";";
    //                src << sp << "e_" << num_elem << ".name = \"" << se.name << "\";";
    //                src << sp << "e_" << num_elem << ".nb_nodes = " << se.nb_nodes() << ";";
    //                src << sp << "e_" << num_elem << ".nb_faces = " << se.nb_faces() << ";";
    //                src << sp << "e_" << num_elem << ".vtk_elements = " << se.vtk_output() << ";";
    //                src << sp << "elem_info.push_back( std::move( e_" << num_elem << " ) );";
    //            }
    //        }
    //    };

    //    struct ToShapeTypes : ComputableTask {
    //        ToShapeTypes( Rc<Task> &&dim_or_shape_types ) : ComputableTask( { std::move( dim_or_shape_types ) } ) {
    //        }

    //        virtual void write_to_stream( std::ostream &os ) const override {
    //            os << "ToShapeTypes";
    //        }

    //        void exec() override {
    //            std::string shape_types = *reinterpret_cast<const std::string *>( children[ 0 ]->output_data );
    //            if ( shape_types[ 0 ] >= '0' && shape_types[ 0 ] <= '9' )
    //                shape_types = default_shape_types( std::stoi( shape_types ) );
    //            make_outputs( TaskOut<std::string>( new std::string( shape_types ) ) );
    //        }

    //        static std::string default_shape_types( int dim ) {
    //            if ( dim == 3 ) return "3S 3E 4S";
    //            if ( dim == 2 ) return "3 4 5";
    //            TODO;
    //            return {};
    //        }
    //    };

    //    task = new GetElementaryPolytopInfoList( new ToShapeTypes( dim_or_shape_types.to_string() ) );
}

ElementaryPolytopTypeSet::ElementaryPolytopTypeSet( const parex::Scalar &dim ) : ElementaryPolytopTypeSet( default_shape_names_for( dim ) ) {
}

parex::Vector<parex::String> ElementaryPolytopTypeSet::default_shape_names_for( const parex::Scalar &dim ) {
    return { new parex::CompiledLambdaInstruction( "get_default_shapes", { dim.variable->get() }, []( parex::Src &src, parex::SrcSet & ) {
        src.compilation_environment.includes << "<parex/utility/TODO.h>";
        src.compilation_environment.includes << "<vector>";
        src.compilation_environment.includes << "<string>";

        src << "std::vector<std::string> *func( int dim ) {\n";
        src << "    if ( dim == 2 ) return new std::vector<std::string>{ \"3\" };\n";
        src << "    if ( dim == 3 ) return new std::vector<std::string>{ \"3S\" };\n";
        src << "    TODO; return nullptr;\n";
        src << "}\n";
    }, 1 ), 1 };
}

} // namespace sdot
