#include "../support/ERROR.h"
#include "../support/P.h"
#include "KernelCode.h"

#include <fstream>

namespace parex {

KernelCode kernel_code;

std::function<void()> KernelCode::operator()( const Kernel &kernel ) {
    auto iter = compilations.find( kernel );
    if ( iter == compilations.end() )
        iter = compilations.insert( iter, { kernel, func( kernel ) } );
    return iter->second.func;
}

KernelCode::Func KernelCode::func( const Kernel &kernel ) {
    // make a new directory
    std::string dir = "objects/" + kernel.name + "/";
    make_CMakeLists( kernel, dir );
    build( kernel, dir );

    return { dir + "build/" + dynalo::to_native_name( "kernel" ), kernel.func };
}

void KernelCode::exec( const std::string &cmd ) {
    std::cout << cmd << std::endl;
    if ( system( cmd.c_str() ) )
        ERROR( "" );
}

void KernelCode::make_CMakeLists( const Kernel &kernel, const std::string &dir ) {
    exec( "mkdir -p " + dir + "build" );

    std::string cmk = dir + "CMakeLists.txt";
    std::ofstream fcmk( cmk );
    fcmk << "project( " << kernel.name << " )\n";
    fcmk << "add_library(kernel SHARED\n";
    fcmk << "    ../../kernels/" << kernel.name << ".cpp\n";
    fcmk << ")\n";
}

void KernelCode::build( const Kernel &/*kernel*/, const std::string &dir ) {
    exec( "cmake -S " + dir + " -B " + dir + "build" );
    exec( "make -C " + dir + "build" );
}

KernelCode::Func::Func( std::string lib, std::string func ) : lib( lib ), func( this->lib.get_function<void(void)>( func ) ) {
}

} // namespace parex
