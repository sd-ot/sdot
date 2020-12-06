#include "GenericArithmeticOperation.h"
#include "../plugins/Src.h"

namespace parex {

GenericArithmeticOperation::GenericArithmeticOperation( std::string name_op, std::vector<Rc<Task>> &&children ) : CompiledTask( std::move( children ) ), name_op( name_op ) {
}

void GenericArithmeticOperation::write_to_stream( std::ostream &os ) const {
    os << "GenericArithmeticOperation<" << name_op << ">";
}

void GenericArithmeticOperation::get_src_content( Src &src, SrcSet &/*sw*/ ) {
    // src.compilation_environment.includes << "<parex/P.h>";
    src.compilation_environment.preliminaries << "using namespace parex;";

    //    src << "    if ( t->move_arg( 0 ) )\n";
    //    src << "        return a " << name_op << "= b, nullptr;\n";
    //    src << "    if ( t->move_arg( 1 ) )\n";
    //    src << "        return b " << name_op << "= a, nullptr;\n";

    src << "template<class T,class U>\n";
    src << "auto " << called_func_name() << "( TaskOut<T> &a, TaskOut<U> &b ) {\n";
    src << "    using R = decltype( *a " << name_op << " *b );\n";
    src << "    return TaskOut<R>( new R( *a " << name_op << " *b ) );\n";
    src << "}\n";
}

} // namespace parex