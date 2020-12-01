#include "internal/SymbolicElementaryPolytop.h"
#include "ElementaryPolytopInfoList.h"

#include <parex/GeneratedSymbolSet.h>
#include <parex/ComputableTask.h>
#include <parex/CppType.h>
#include <parex/TODO.h>

namespace sdot {

ElementaryPolytopInfoList::ElementaryPolytopInfoList( const Value &dim_or_shape_types ) {
    struct GetElementaryPolytopInfoList : ComputableTask {
        GetElementaryPolytopInfoList( Rc<Task> &&shape_types ) : ComputableTask( { std::move( shape_types ) } ) {
        }

        virtual void write_to_stream( std::ostream &os ) const override {
            os << "GetElementaryPolytopInfoList";
        }

        virtual void exec() override {
            // set the output type
            output_type = type_factory().reg_cpp_type( "ElementaryPolytopInfoListContent", []( CppType &ct ) {
                ct.compilation_environment.includes << "<sdot/geometry/internal/ElementaryPolytopInfoListContent.h>";
                ct.compilation_environment.include_directories << SDOT_DIR "/src";
            } );

            // summary
            std::string shape_types = *reinterpret_cast<const std::string *>( children[ 0 ]->output_data );

            // find or create lib
            static GeneratedSymbolSet gls;
            auto *func = gls.get_symbol<void( ComputableTask *)>( [&]( SrcSet &sw ) {
                Src &src = sw.src( "get_ElementaryPolytopInfoList.cpp" );

                // src.cpp_flags << "-std=c++17" << "-g3";
                src.compilation_environment.includes << "<parex/ComputableTask.h>";
                output_type->add_needs_in( src );

                src << "namespace {\n";
                src << "struct Epil : ElementaryPolytopInfoListContent {\n";
                src << "    Epil() {";
                write_epil_ctor( src, shape_types, "\n        " );
                src << "    }\n";
                src << "};\n";
                src << "}\n";

                src << "\n";
                src << "extern \"C\" void exported( ComputableTask *task ) {\n";
                src << "    static Epil res;\n";
                src << "    task->output_data = &res;\n";
                src << "    task->output_own = false;\n";
                src << "}\n";
            }, shape_types );

            // execute the generated function to get the output_data
            func( this );
        }

        void write_epil_ctor( Src &src, const std::string &shape_types, const std::string &sp ) {
            std::vector<SymbolicElementaryPolytop> lst;
            std::istringstream ss( shape_types );
            for( std::string name; ss >> name; )
                lst.push_back( name );

            src << sp << "default_dim = 2;";

            for( std::size_t num_elem = 0; num_elem < lst.size(); ++num_elem ) {
                SymbolicElementaryPolytop &se = lst[ num_elem ];
                src << sp << "";
                src << sp << "ElementaryPolytopInfo e_" << num_elem << ";";
                src << sp << "e_" << num_elem << ".name = \"" << se.name << "\";";
                src << sp << "e_" << num_elem << ".nb_nodes = " << se.nb_nodes() << ";";
                src << sp << "e_" << num_elem << ".nb_faces = " << se.nb_faces() << ";";
                src << sp << "elem_info.push_back( std::move( e_" << num_elem << " ) );";
            }
        }
    };

    struct ToShapeTypes : ComputableTask {
        ToShapeTypes( Rc<Task> &&dim_or_shape_types ) : ComputableTask( { std::move( dim_or_shape_types ) } ) {
        }

        virtual void write_to_stream( std::ostream &os ) const override {
            os << "ToShapeTypes";
        }

        void exec() override {
            std::string shape_types = *reinterpret_cast<const std::string *>( children[ 0 ]->output_data );
            if ( shape_types[ 0 ] >= '0' && shape_types[ 0 ] <= '9' )
                shape_types = default_shape_types( std::stoi( shape_types ) );
            make_outputs( TaskOut<std::string>( new std::string( shape_types ) ) );
        }

        static std::string default_shape_types( int dim ) {
            if ( dim == 3 ) return "3S 3E 4S";
            if ( dim == 2 ) return "3 4 5";
            TODO;
            return {};
        }
    };

    task = new GetElementaryPolytopInfoList( new ToShapeTypes( dim_or_shape_types.to_string() ) );
}

void ElementaryPolytopInfoList::write_to_stream( std::ostream &os ) const {
    os << Value( task );
}

} // namespace sdot
