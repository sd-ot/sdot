#include "support/cstr_encode.h"
#include "support/url_encode.h"
#include "support/ERROR.h"
#include "support/P.h"
#include "KernelCode.h"

#include <algorithm>
#include <fstream>

// cpu_features
#include <cpu_features/cpu_features_macros.h>

#if defined(CPU_FEATURES_ARCH_X86)
    #include <cpu_features/cpuinfo_x86.h>
#elif defined(CPU_FEATURES_ARCH_ARM)
    #include <cpu_features/cpuinfo_arm.h>
#elif defined(CPU_FEATURES_ARCH_AARCH64)
    #include <cpu_features/cpuinfo_aarch64.h>
#elif defined(CPU_FEATURES_ARCH_MIPS)
    #include <cpu_features/cpuinfo_mips.h>
#elif defined(CPU_FEATURES_ARCH_PPC)
    #include <cpu_features/cpuinfo_ppc.h>
#endif

namespace parex {

KernelCode kernel_code;

KernelCode::KernelCode() {
    init_default_flags();

    src_heads[ "ostream" ].push_back( "using std::ostream;" );

    src_heads[ "SI32"    ].push_back( "using SI32 = std::int32_t;" );
    includes [ "SI32"    ].push_back( "<cstdint>" );

    src_heads[ "PI64"    ].push_back( "using PI64 = std::uint64_t;" );
    includes [ "PI64"    ].push_back( "<cstdint>" );

    src_heads[ "FP64"    ].push_back( "using FP64 = double;" );
    src_heads[ "FP32"    ].push_back( "using FP32 = float;" );

    includes [ "Tensor"  ].push_back( "<parex/containers/Tensor.h>" );
    includes [ "Vec"     ].push_back( "<parex/containers/Vec.h>" );

    include_directories.push_back( PAREX_DIR "/src/parex/kernels" );
}

void KernelCode::add_include_dir( std::string name ) {
    if ( std::find( include_directories.begin(), include_directories.end(), name ) == include_directories.end() )
        include_directories.push_back( name );
}

KernelCode::Func KernelCode::func( const Kernel &kernel, const std::vector<std::string> &input_types ) {
    Src src{ kernel, input_types, {} };
    auto iter = code.find( src );
    if ( iter == code.end() )
        iter = code.insert( iter, { src, make_code( kernel, input_types ) } );
    return iter->second.func;
}

void KernelCode::init_default_flags() {
    #if defined(CPU_FEATURES_ARCH_X86)
    cpu_features::X86Info ci = cpu_features::GetX86Info();
    if ( ci.features.avx512f )
        cpu_config = "avx512";
    else if ( ci.features.avx2 )
        cpu_config = "avx2";
    else if ( ci.features.sse2 )
        cpu_config = "sse2";
    #else
    TODO;
    #endif
}

KernelCode::Code KernelCode::make_code( const Kernel &kernel, const std::vector<std::string> &input_types ) {
    // directory name
    std::string dir = "objects/" + url_encode( kernel.name ) + "/" + cpu_config + "/";
    if ( input_types.size() ) {
        for( std::size_t i = 0; i < input_types.size(); ++i ) {
            std::string url = url_encode( input_types[ i ] );
            dir += ( i ? "_" : "" ) + std::to_string( url.size() ) + "_" + url;
        }
    } else
        dir += "no_input";
    dir += "/";

    // directory creation
    exec( "mkdir -p " + dir + "build" );

    // base name param
    std::string bname = kernel.name, param;
    auto pp = kernel.name.find( '(' );
    bool gc = pp != kernel.name.npos;
    if ( gc ) {
        ASSERT( kernel.name.back() == ')', "" );
        param = kernel.name.substr( pp + 1, kernel.name.size() - pp - 2 );
        bname = kernel.name.substr( 0, pp );
        gen_code( dir, bname, param );
    }

    // create a CmakeLists.txt and a kernel.cpp file.
    make_kernel_cpp( dir, bname, input_types, kernel.task_as_arg, gc );
    make_cmake_lists( dir, bname, {} );

    // build the library
    build_kernel( dir );

    // load ir
    Code res;
    res.lib = std::make_unique<dynalo::library>( dir + "build/" + dynalo::to_native_name( bname ) );
    res.func = res.lib->get_function<void(Task*,void **)>( "kernel_wrapper" );
    return res;
}

bool KernelCode::gen_code( const std::string &dir, const std::string &bname, const std::string &param ) {
    std::string gdir = dir + "gen/";
    exec( "mkdir -p " + gdir + "build" );

    // CMakeLists.txt
    std::ofstream fcmk( gdir + "CMakeLists.txt" );
    fcmk << "cmake_minimum_required(VERSION 3.0)\n";
    fcmk << "project(" << bname << "_generator)\n";

    fcmk << "\n";
    fcmk << "add_executable(generator\n";
    fcmk << "    generator.cpp\n";
    fcmk << ")\n";

    fcmk << "\n";
    fcmk << "target_compile_options(generator PRIVATE -march=native -O3 -g3)\n";

    fcmk << "\n";
    fcmk << "target_include_directories(generator PRIVATE " << PAREX_DIR "/src" << ")\n";
    for( std::string include_directory : include_directories )
        fcmk << "target_include_directories(generator PRIVATE " << include_directory << ")\n";

    fcmk.close();

    // generator.cpp
    std::ofstream fcpp( gdir + "generator.cpp" );
    fcpp << "#include <" << bname << ".h>\n";
    fcpp << "#include <fstream>\n";

    fcpp << "\n";
    fcpp << "int main( int, char **argv ) {\n";
    fcpp << "    std::ofstream fout( \"" << dir << bname << ".h\" );\n";
    fcpp << "    " << bname << "( fout, \"" << bname << "\", \"" << cstr_encode( param ) << "\" );\n";
    fcpp << "}\n";

    fcpp.close();

    //
    build_kernel( gdir );
    exec( gdir + "build/generator" );

    return true;
}

void KernelCode::exec( const std::string &cmd ) {
    // std::cout << cmd << std::endl;
    if ( system( cmd.c_str() ) ) {
        ERROR( "Error in cmd: {}", cmd );
    }
}

void KernelCode::make_cmake_lists( const std::string &dir, const std::string &name, const std::vector<std::string> &/*flags*/ ) {
    std::ofstream fcmk( dir + "CMakeLists.txt" );
    fcmk << "cmake_minimum_required(VERSION 3.0)\n";
    fcmk << "project( " << name << " )\n";

    fcmk << "\n";
    fcmk << "add_library(" << name << " SHARED\n";
    fcmk << "    kernel.cpp\n";
    fcmk << ")\n";

    fcmk << "\n";
    fcmk << "target_compile_options(" << name << " PRIVATE -march=native -O3 -g3)\n";

    fcmk << "\n";
    fcmk << "target_include_directories(" << name << " PRIVATE " << PAREX_DIR "/src" << ")\n";
    for( std::string include_directory : include_directories )
        fcmk << "target_include_directories(" << name << " PRIVATE " << include_directory << ")\n";
}

void KernelCode::make_kernel_cpp( const std::string &dir, const std::string &name, const std::vector<std::string> &input_types, bool task_as_arg, bool local_inc ) {
    std::ofstream fcpp( dir + "kernel.cpp" );

    // prerequisites for the types
    std::set<std::string> include_set, src_head_set, seen_types;
    std::ostringstream includes, src_heads;
    for( const std::string &input_type : input_types )
        get_prereq_req( includes, src_heads, include_set, src_head_set, seen_types, input_type );

    // header(s) and typedefs
    fcpp << includes.str();

    fcpp << "\n";
    fcpp << "#include <parex/TaskRef.h>\n";
    fcpp << "#include " << ( local_inc ? '"' : '<' ) << name << ".h" << ( local_inc ? '"' : '>' ) << "\n";

    fcpp << "\n";
    fcpp << src_heads.str();

    //
    fcpp << "\n";
    fcpp << "namespace {\n";
    fcpp << "    struct KernelWrapper {\n";
    fcpp << "        auto operator()( parex::Task *" << ( task_as_arg ? "task" : "/*task*/" ) << ", void **data ) const {\n";
    fcpp << "            return " << name << "(\n";
    if ( task_as_arg )
        fcpp << "                task" << ( input_types.size() ? "," : "" ) << "\n";
    for( std::size_t i = 0; i < input_types.size(); ++i )
        fcpp << "                *reinterpret_cast<" << input_types[ i ] << "*>( data[ " << i << " ] )" << ( i + 1 < input_types.size() ? "," : "" ) << "\n";
    fcpp << "            );\n";
    fcpp << "        }\n";
    fcpp << "    };\n";
    fcpp << "}\n";

    fcpp << "\n";
    fcpp << "extern \"C\" void kernel_wrapper( parex::Task *task, void **data ) {\n";
    fcpp << "    task->run( KernelWrapper(), data );\n";
    fcpp << "}\n";
}

void KernelCode::get_prereq_req( std::ostream &includes_os, std::ostream &src_heads_os, std::set<std::string> &includes_set, std::set<std::string> &src_heads_set, std::set<std::string> &seen_types, const std::string &type ) {
    if ( seen_types.count( type ) )
        return;
    seen_types.insert( type );

    // something like X<Y,...> ?
    std::size_t s = type.find( '<' );
    if ( s != type.npos ) {
        get_prereq_req( includes_os, src_heads_os, includes_set, src_heads_set, seen_types, type.substr( 0, s ) );

        for( std::size_t b = s + 1, c = b, n = 0; c < type.size(); ++c ) {
            switch ( type[ c ] ) {
            case '>': if ( n-- == 0 ) { get_prereq_req( includes_os, src_heads_os, includes_set, src_heads_set, seen_types, type.substr( b, c - b ) ); b = c + 1; } break;
            case ',': if ( n == 0 ) { get_prereq_req( includes_os, src_heads_os, includes_set, src_heads_set, seen_types, type.substr( b, c - b ) ); b = c + 1; } break;
            case '<': ++n; break;
            default: break;
            }
        }
    }


    // include
    auto iter_inc = includes.find( type );
    if ( iter_inc != includes.end() )
        for( const std::string &h : iter_inc->second )
            if ( includes_set.insert( h ).second )
                includes_os << "#include " << h << "\n";

    // src_head
    auto iter_hea = src_heads.find( type );
    if ( iter_hea != src_heads.end() )
        for( const std::string &h : iter_hea->second )
            if ( src_heads_set.insert( h ).second )
                src_heads_os << h << "\n";
}

void KernelCode::build_kernel( const std::string &dir ) {
    exec( "cmake -S " + dir + " -B " + dir + "build > /dev/null" );
    exec( "cmake --build " + dir + "build > /dev/null" ); // --target install
}

} // namespace parex
