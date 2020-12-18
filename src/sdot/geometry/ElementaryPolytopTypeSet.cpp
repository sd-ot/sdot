#include <parex/instructions/CompiledLambdaInstruction.h>
#include <parex/compilation/GeneratedSymbolSet.h>
#include <parex/data/CompiledType.h>
#include <parex/utility/va_string.h>
#include <parex/utility/ASSERT.h>
#include <parex/utility/P.h>

#include "internal/SymbolicElementaryPolytop.h"
#include "ElementaryPolytopTypeSet.h"


namespace sdot {

ElementaryPolytopTypeSet::ElementaryPolytopTypeSet( const parex::Vector<parex::String> &shape_names ) {
    struct GetElementaryPolytopCaracList : parex::CompiledInstruction {
    public:
        using parex::CompiledInstruction::CompiledInstruction;

        virtual void prepare( parex::TypeFactory *tf, parex::SchedulerSession */*ss*/ ) override {
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

        virtual void get_src_content( parex::Src &src, parex::SrcSet &/*sw*/, parex::TypeFactory *tf ) const override {
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

        virtual std::string summary() const override {
            return parex::va_string( "GetElementaryPolytopCaracList {}", shape_names() );
        }

        const std::vector<std::string> &shape_names() const {
            ASSERT( slots[ 0 ].input.first_slot() && slots[ 0 ].input.first_slot()->data, "" );
            return *reinterpret_cast<std::vector<std::string> *>( slots[ 0 ].input.data_ptr() );
        }

        std::string type_name() const {
            std::string res = "ElementaryPolytopCaracList";
            for( const std::string &shape_name : shape_names() )
                res += "_" + shape_name;
            return res;
        }

        void write_carac( parex::Src &src, const SymbolicElementaryPolytop &se ) const {
            std::string vn = "ep_" + se.name;

            src << "    sdot::ElementaryPolytopCarac " + vn + ";\n";
            src << "    " + vn + ".vtk_elements = " << se.vtk_output() << ";\n";
            src << "    " + vn + ".nb_nodes = " << se.nb_nodes() << ";\n";
            src << "    " + vn + ".nb_faces = " << se.nb_faces() << ";\n";
            src << "    " + vn + ".name = \"" << se.name << "\";\n";
            src << "    res.elements.push_back( std::move( " + vn + " ) );\n";
        }
    };
    variable = new parex::Variable( new GetElementaryPolytopCaracList( "GetElementaryPolytopCaracList", { shape_names.to<std::vector<std::string>>().variable->get() }, 1 ), 1 );
}

ElementaryPolytopTypeSet::ElementaryPolytopTypeSet( const parex::Scalar &dim ) : ElementaryPolytopTypeSet( default_shape_names_for( dim ) ) {
}

parex::Vector<parex::String> ElementaryPolytopTypeSet::default_shape_names_for( const parex::Scalar &dim ) {
    return { new parex::CompiledLambdaInstruction( "get_default_shapes", { dim.variable->get() }, []( parex::Src &src, parex::SrcSet &, parex::TypeFactory * ) {
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
