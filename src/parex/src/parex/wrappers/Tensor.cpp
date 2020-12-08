#include "../tasks/GenericArithmeticOperation.h"
#include "../tasks/CompiledIncludeTask.h"
#include "../utility/TODO.h"
#include "../utility/P.h"
#include "Tensor.h"

namespace parex {

Tensor::Tensor( Task *t ) : TaskWrapper( t ) {}

Tensor Tensor::from_function( std::function<void(Src &,SrcSet &)> &&code, const Vector &shape, ListOfTask &&args, const String &type, Memory *memory ) {
    struct FromFunction : CompiledTask {
        /**/ FromFunction( std::function<void(Src &,SrcSet &)> &&code, std::vector<Rc<Task>> &&ch ) : CompiledTask( "FromFunction", std::move( ch ) ), code( std::move( code ) ) {
        }

        virtual void get_src_content( Src &src, SrcSet &sw ) override {
            Type *type = type_factory( *reinterpret_cast<std::string *>( children[ children.size() - 1 ]->output.data ) );
            int dim = *reinterpret_cast<int *>( children[ children.size() - 2 ]->output.data );
            src.compilation_environment.preliminaries << "using namespace parex;";
            src.compilation_environment.includes << "<parex/containers/gtensor.h>";
            src.compilation_environment += type->compilation_environment;

            std::string gt = "gtensor<" + type->name + "," + std::to_string( dim ) + ",Allocator>";

            src << "template<class Shape,class Allocator>\n";
            src << "TaskOut<" + gt + "> " << called_func_name() << "( const TaskOut<Shape> &shape, TaskOut<Allocator> &allocator, TaskOut<int> &, TaskOut<std::string> & ) {\n";
            src << "    using I = typename " + gt + "::I;\n";
            src << "    using T = " + type->name + ";\n";
            src << "    \n";

            src << "    auto res = " + gt + "::empty( &*allocator, {";
            for( int i = 0; i < dim; ++i )
                src << ( i ? ", " : " " ) << "I( shape->at( " << i << " ) )";
            src << " } );\n";

            // TODO: simd, gpu
            src << "    T *data = res.data();\n";
            src << "    res.for_each_offset_and_index( [&]( I offset";
            for( int i = 0; i < dim; ++i )
                src << ", I " << ( dim < 7 ? std::string( 1, "ijklmno"[ i ] ) : "ind_" + std::to_string( i ) );
            src << " ) {\n";
            src << "        data[ offset ] = ";
            code( src, sw );
            src << ";\n";
            src << "    } );\n";

            src << "    return new " + gt + "( std::move( res ) );\n";
            src << "}\n";
        }

        std::function<void(Src &,SrcSet &)> code;
    };

    args << shape.task << memory->allocator_as_task() << shape.size().conv_to( TypeInfo<int>::name() ) << type.to_string();
    return new FromFunction( std::move( code ), std::move( args.tasks ) );
}

Tensor Tensor::from_function( const std::string &code, const Vector &shape, ListOfTask &&args, const String &type, Memory *memory ) {
    return from_function( [&]( Src &src, SrcSet & ) { src << code; }, shape, std::move( args ), type, memory );
}

Tensor Tensor::operator+( const Tensor &that ) const { return new GenericArithmeticOperation( "+", { task, that.task } ); }
Tensor Tensor::operator-( const Tensor &that ) const { return new GenericArithmeticOperation( "-", { task, that.task } ); }
Tensor Tensor::operator*( const Tensor &that ) const { return new GenericArithmeticOperation( "*", { task, that.task } ); }
Tensor Tensor::operator/( const Tensor &that ) const { return new GenericArithmeticOperation( "/", { task, that.task } ); }

Tensor &Tensor::operator+=( const Tensor &that ) { task = new GenericArithmeticOperation( "+", { task, that.task } ); return *this; }
Tensor &Tensor::operator-=( const Tensor &that ) { task = new GenericArithmeticOperation( "-", { task, that.task } ); return *this; }
Tensor &Tensor::operator*=( const Tensor &that ) { task = new GenericArithmeticOperation( "*", { task, that.task } ); return *this; }
Tensor &Tensor::operator/=( const Tensor &that ) { task = new GenericArithmeticOperation( "/", { task, that.task } ); return *this; }

} // namespace parex
