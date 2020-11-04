#include "../support/ERROR.h"
#include "KernelCode.h"

#include <fstream>

namespace parex {

KernelCode kernel_code;

KernelCode::Func KernelCode::func( const Kernel &kernel, const std::vector<std::string> &input_types ) {
    Src src{ kernel, input_types, {} };
    auto iter = code.find( src );
    if ( iter == code.end() )
        iter = code.insert( iter, { src, make_code( kernel, input_types ) } );
    return iter->second.func;
}

KernelCode::Code KernelCode::make_code( const Kernel &kernel, const std::vector<std::string> &input_types ) {
    std::string dir = "objects/" + kernel.name + "/";
    exec( "mkdir -p " + dir + "build" );
    make_cmake_lists( dir, name, flags );
    make_cpp( dir,  );
    build_kernel( dir );

    Code res;
    res.lib = std::make_unique<dynalo::library>( dir + "build/" + dynalo::to_native_name( "kernel" ) );
    res.func = res.lib->get_function<void(void **)>( "kernel_wrapper" );
    return res;
}

void KernelCode::exec( const std::string &cmd ) {
    std::cout << cmd << std::endl;
    if ( system( cmd.c_str() ) )
        ERROR( "" );
}

void KernelCode::make_cmake_lists( const std::string &dir, const std::string &name, const std::vector<std::string> &/*flags*/ ) {
    std::string cmk = dir + "CMakeLists.txt";

    std::ofstream fcmk( cmk );
    fcmk << "project( " << name << " )\n";
    fcmk << "add_library(kernel SHARED\n";
    fcmk << "    ../../kernels/" << name << ".cpp\n";
    fcmk << ")\n";
}

void KernelCode::build_kernel( const std::string &dir ) {
    exec( "cmake -S " + dir + " -B " + dir + "build" );
    exec( "cmake --build " + dir + "build" ); // --target install
}

void KernelCode::make_cpp( const std::string &dir, const Kernel &kernel ) {
    std::string cpp = dir + "CMakeLists.txt";

    std::ofstream fpp( cpp );
    fcmk << "project( " << name << " )\n";
    fcmk << "add_library(kernel SHARED\n";
    fcmk << "    ../../kernels/" << name << ".cpp\n";
    fcmk << ")\n";
}

} // namespace parex
