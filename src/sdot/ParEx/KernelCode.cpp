#include "internal/url_encode.h"
#include "../support/ERROR.h"
#include "KernelCode.h"

#include <fstream>
#include <set>

namespace parex {

KernelCode kernel_code;

KernelCode::KernelCode() {
    src_heads[ "ostream" ].push_back( "using std::ostream;" );

    src_heads[ "SI32" ].push_back( "using SI32 = std::int32_t;" );
    includes [ "SI32" ].push_back( "<cstdint>" );

    src_heads[ "PI64" ].push_back( "using PI64 = std::uint64_t;" );
    includes [ "PI64" ].push_back( "<cstdint>" );

    src_heads[ "FP64" ].push_back( "using FP64 = double;" );
    src_heads[ "FP32" ].push_back( "using FP32 = float;" );
}

KernelCode::Func KernelCode::func( const Kernel &kernel, const std::vector<std::string> &input_types ) {
    Src src{ kernel, input_types, {} };
    auto iter = code.find( src );
    if ( iter == code.end() )
        iter = code.insert( iter, { src, make_code( kernel, input_types ) } );
    return iter->second.func;
}

KernelCode::Code KernelCode::make_code( const Kernel &kernel, const std::vector<std::string> &input_types ) {
    // directory name
    std::string dir = "objects/" + kernel.name + "/";
    for( std::string input_type : input_types ) {
        std::string url = urlencode( input_type );
        dir += "_" + std::to_string( url.size() ) + "_" + url;
    }
    dir += "/";

    // directory creation
    exec( "mkdir -p " + dir + "build" );

    // create a CmakeLists.txt and a kernel.cpp file.
    make_cmake_lists( dir, kernel.name, {} );
    make_kernel_cpp( dir, kernel.name, input_types );

    // build the library
    build_kernel( dir );

    // load ir
    Code res;
    res.lib = std::make_unique<dynalo::library>( dir + "build/" + dynalo::to_native_name( kernel.name ) );
    res.func = res.lib->get_function<void(Task*,void **)>( "kernel_wrapper" );
    return res;
}

void KernelCode::exec( const std::string &cmd ) {
    // std::cout << cmd << std::endl;
    if ( system( cmd.c_str() ) )
        ERROR( "" );
}

void KernelCode::make_cmake_lists( const std::string &dir, const std::string &name, const std::vector<std::string> &/*flags*/ ) {
    std::ofstream fcmk( dir + "CMakeLists.txt" );
    fcmk << "project( " << name << " )\n";
    fcmk << "\n";
    fcmk << "add_library(" << name << " SHARED\n";
    fcmk << "    kernel.cpp\n";
    fcmk << ")\n";
}

void KernelCode::make_kernel_cpp( const std::string &dir, const std::string &name, const std::vector<std::string> &input_types ) {
    std::ofstream fcpp( dir + "kernel.cpp" );

    // needed includes
    std::set<std::string> in;
    for( const std::string &input_type : input_types ) {
        auto iter = includes.find( input_type );
        if ( iter != includes.end() )
            for( const std::string &h : iter->second )
                if ( in.insert( h ).second )
                    fcpp << "#include " << h << "\n";
    }

    fcpp << "#include \"../../../src/sdot/ParEx/TaskRef.h\"\n";
    fcpp << "#include \"../../../kernels/" << name << ".h\"\n";

    // needed src_heads
    std::set<std::string> sh;
    for( const std::string &input_type : input_types ) {
        auto iter = src_heads.find( input_type );
        if ( iter != src_heads.end() )
            for( const std::string &h : iter->second )
                if ( sh.insert( h ).second )
                    fcpp << h << "\n";
    }

    //
    fcpp << "\n";
    fcpp << "extern \"C\" void kernel_wrapper( parex::Task *task, void **data ) {\n";
    fcpp << "    auto res = " << name << "(\n";
    for( std::size_t i = 0; i < input_types.size(); ++i )
        fcpp << "        *reinterpret_cast<" << input_types[ i ] << "*>( data[ " << i << " ] )" << ( i + 1 < input_types.size() ? "," : "" ) << "\n";
    fcpp << "    );\n";
    fcpp << "    task->output_type = parex::type_name( res );\n";
    fcpp << "    task->output_data = res;\n";
    fcpp << "}\n";
}

void KernelCode::build_kernel( const std::string &dir ) {
    exec( "cmake -S " + dir + " -B " + dir + "build > /dev/null" );
    exec( "cmake --build " + dir + "build > /dev/null" ); // --target install
}

} // namespace parex
