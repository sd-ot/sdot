#include "GenericArithmeticOperation.h"
#include "../Src.h"

GenericArithmeticOperation::GenericArithmeticOperation( std::string name_op, std::vector<Rc<Task>> &&children ) : CompiledTask( std::move( children ) ), name_op( name_op ) {
}

void GenericArithmeticOperation::write_to_stream( std::ostream &os ) const {
    os << "GenericArithmeticOperation<" << name_op << ">";
}

void GenericArithmeticOperation::get_src_content( Src &src, SrcSet &/*sw*/ ) {
    src.includes << "<parex/TaskOut.h>";
    src.includes << "<parex/P.h>";

    src << "template<class T>\n";
    src << "TaskOut<T> " << func_name() << "( TaskOut<T> &a, TaskOut<T> &b ) {\n";
    //    src << "    if ( t->move_arg( 0 ) )\n";
    //    src << "        return a " << name_op << "= b, nullptr;\n";
    //    src << "    if ( t->move_arg( 1 ) )\n";
    //    src << "        return b " << name_op << "= a, nullptr;\n";
    src << "    return new T( *a " << name_op << " *b );\n";
    src << "}\n";
}
