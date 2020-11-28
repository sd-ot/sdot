#include "KernelWithCompiledCode.h"
#include "TypeInfoFactory.h"
#include "ArchId.h"
#include "Task.h"
#include "TODO.h"

#include <sstream>
#include <set>

KernelWithCompiledCode::KernelWithCompiledCode() : ckc( this ) {
}

void KernelWithCompiledCode::exec( Task *task ) const {
    auto *func = ckc.get_func( task );
    func( task );
}

void KernelWithCompiledCode::make_srcs( SrcSet &ff ) const {
    std::vector<TypeInfo *> tis = type_infos( ff.parameters );
    ff.default_cpp_flags = "-march=native -O3 -g3 -std=c++17";

    // includes
    std::set<std::string> si;
    auto add_include = [&]( const std::string &include ) {
        if ( si.count( include ) )
            return;
        si.insert( include );
        ff << "#include " << include << "\n";
    };
    for( const auto &include : ff.default_includes )
        add_include( include );
    for( TypeInfo *type_info : tis )
        type_info->get_includes( add_include );

    // preliminaries
    std::set<std::string> sp;
    auto add_preliminary = [&]( const std::string &preliminary ) {
        if ( sp.count( preliminary ) )
            return;
        sp.insert( preliminary );
        ff << preliminary << "\n";
    };
    for( TypeInfo *type_info : tis )
        type_info->get_preliminaries( add_preliminary );

    // func
    ff << "#include <parex/TypedVariant.h>\n";
    ff << "#include <parex/Task.h>\n";
    ff << "\n";
    ff << "namespace {\n";
    ff << "    struct KernelWrapper {\n";
    ff << "        auto operator()( Task *task ) const {\n";
    for( std::size_t num_child = 0; num_child < tis.size(); ++num_child )
        ff << "            TypedVariant<" << tis[ num_child ]->cpp_name() << "> arg_" << num_child << "( *task->children[ " << num_child << " ]->output->variants[ 0 ] );\n";
    ff << "            return " << func_name( ff.parameters ) << "(";
    for( std::size_t num_child = 0; num_child < tis.size(); ++num_child )
        ff << ( num_child + 1 < tis.size() ? ", " : " " ) << "arg_" << num_child;
    ff << " );\n";
    ff << "        }\n";
    ff << "    };\n";
    ff << "}\n";
    ff << "\n";
    ff << "extern \"C\" void " << ff.symbol_name << "( Task *task ) {\n";
    ff << "    task->run_kernel_wrapper( KernelWrapper() );\n";
    ff << "}\n";
}

std::string KernelWithCompiledCode::kernel_parameters( const Task *task ) {
    std::ostringstream ss;
    ss << task->children.size();
    for( const RcPtr<Task> &child : task->children )
        ss << "\n" << child->output->variants[ 0 ]->type.name;

    ss << "\n";
    get_summary( ss );

    ss << "\n";
    ArchId arch_id;
    ss << arch_id.name;

    return ss.str();
}

std::vector<TypeInfo *> KernelWithCompiledCode::type_infos( const std::string &kernel_parameters ) const {
    std::istringstream ss( kernel_parameters );
    std::string line;
    std::getline( ss, line );
    std::size_t nb_inputs = std::stoi( line );

    std::vector<TypeInfo *> res( nb_inputs );
    for( std::size_t i = 0; i < nb_inputs; ++i ) {
        std::getline( ss, line );
        res[ i ] = type_info_factory( line );
    }

    return res;
}
