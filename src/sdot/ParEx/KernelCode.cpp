#include "../support/ERROR.h"
#include "KernelCode.h"

#include <fstream>

namespace parex {

KernelCode kernel_code;

 KernelCode::Func KernelCode::func( const Kernel &kernel ) {
    auto iter = funcs.find( kernel );
    if ( iter == funcs.end() )
        iter = funcs.insert( iter, { kernel, make_func( kernel ) } );
    return iter->second;
}

KernelCode::Lib *KernelCode::lib( const std::string &name, std::vector<std::string> flags ) {
    Src src{ name, flags };
    auto iter = libs.find( src );
    if ( iter == libs.end() )
        iter = libs.insert( iter, { src, make_lib( name, flags ) } );
    return &iter->second;
}

KernelCode::Func KernelCode::make_func( const Kernel &kernel ) {
    Lib *l = lib( kernel.name, {} );
    return l->get_function<void(void **)>( "kernel_wrapper" );
}

KernelCode::Lib KernelCode::make_lib( const std::string &name, const std::vector<std::string> &flags ) {
    std::string dir = "objects/" + name + "/";
    make_cmake_lists( dir, name, flags );
    build_kernel( dir );

    return Lib{ dir + "build/" + dynalo::to_native_name( "kernel" ) };
}

void KernelCode::exec( const std::string &cmd ) {
    std::cout << cmd << std::endl;
    if ( system( cmd.c_str() ) )
        ERROR( "" );
}

void KernelCode::make_cmake_lists( const std::string &dir, const std::string &name, const std::vector<std::string> &/*flags*/ ) {
    std::string cmk = dir + "CMakeLists.txt";
    exec( "mkdir -p " + dir + "build" );

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

} // namespace parex
