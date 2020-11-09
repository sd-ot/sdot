#include "support/cstr_encode.h"
#include "support/url_encode.h"
#include "support/ERROR.h"
#include "support/TODO.h"
#include "support/P.h"
#include "KernelCode.h"

#include <algorithm>
#include <sstream>
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
    object_dir = "objects";

    include_directories.push_back( path( PAREX_DIR ) / "src" / "parex" / "kernels" );
    init_default_flags();
    init_base_types();
}

void KernelCode::add_include_dir( path name ) {
    if ( std::find( include_directories.begin(), include_directories.end(), name ) == include_directories.end() )
        include_directories.push_back( name );
}

KernelCode::Func KernelCode::func( const Kernel &kernel, const std::vector<std::string> &input_types ) {
    // summary
    std::ostringstream ss;
    ss << kernel.name << "\n";
    ss << input_types.size() << "\n";
    for( const std::string &input_type : input_types )
        ss << input_type << "\n";
    ss << cpu_config << "\n";

    // not in the map ?
    auto iter = code.find( ss.str() );
    if ( iter == code.end() )
        iter = code.insert( iter, { ss.str(), load_or_make_code( ss.str(), kernel, input_types ) } );

    // => in memory
    return iter->second.func;
}

KernelCode::Code KernelCode::load_or_make_code( const std::string &kstr, const Kernel &kernel, const std::vector<std::string> &input_types ) {
    // already in the filesystem ?
    std::hash<std::string> hasher;
    std::size_t hash = hasher( kstr );

    for( ; ; ++hash ) {
        std::string shash = std::to_string( hash );
        path pinfo = object_dir / ( shash + ".info" );

        // no file with corresponding name => create a new lib/info pair
        if ( ! std::filesystem::exists( pinfo ) ) {
            make_code( shash, kstr, kernel, input_types );
            return load_code( shash );
        }

        // else, if info is good, return the lib
        std::ifstream finfo( pinfo );
        std::ostringstream sinfo;
        sinfo << finfo.rdbuf();
        if ( sinfo.str() == kstr )
            return load_code( shash );
    }
}

KernelCode::Code KernelCode::load_code( const std::string &shash ) {
    path p = object_dir / "lib" / dynalo::to_native_name( shash );

    Code res;
    res.lib = std::make_unique<dynalo::library>( p.string() );
    res.func = res.lib->get_function<void(Task*,void **)>( "kernel_wrapper" );
    return res;

}

void KernelCode::make_code( const std::string &shash, const std::string &kstr, const Kernel &kernel, const std::vector<std::string> &input_types ) {
    // lib/lib___.so
    TmpDir tmp_dir;
    make_lib( tmp_dir, shash, kernel, input_types );

    // ___.info
    std::ofstream fkstr( object_dir / ( shash + ".info" ) );
    fkstr << kstr;
}

void KernelCode::make_lib( TmpDir &tmp_dir, const std::string &shash, const Kernel &kernel, const std::vector<std::string> &input_types ) {
    make_cpp( tmp_dir, kernel, input_types );
    make_cmk( tmp_dir, shash );
    build( tmp_dir.p, " --target install" );
}

void KernelCode::make_cmk( TmpDir &tmp_dir, const std::string &shash ) {
    std::ofstream fcmk( tmp_dir.p / "CMakeLists.txt" );

    fcmk << "cmake_minimum_required(VERSION 3.0)\n";
    fcmk << "project(kernel)\n";

    fcmk << "\n";
    fcmk << "add_library(" << shash << " SHARED\n";
    fcmk << "    kernel.cpp\n";
    fcmk << ")\n";

    fcmk << "\n";
    fcmk << "install(TARGETS " << shash << ")\n";

    fcmk << "\n";
    fcmk << "target_compile_options(" << shash << " PRIVATE -march=native -O3 -g3)\n";

    fcmk << "\n";
    fcmk << "target_include_directories(" << shash << " PRIVATE " << PAREX_DIR "/src" << ")\n";
    for( std::string include_directory : include_directories )
        fcmk << "target_include_directories(" << shash << " PRIVATE " << include_directory << ")\n";
}

void KernelCode::make_cpp( TmpDir &tmp_dir, const Kernel &kernel, const std::vector<std::string> &input_types ) {
    std::ofstream fcpp( tmp_dir.p / "kernel.cpp" );

    // generated kernel ?
    std::string bname = kernel.name;
    std::string param;
    auto pp = kernel.name.find( '(' );
    bool generated = pp != kernel.name.npos;
    if ( generated ) {
        ASSERT( kernel.name.back() == ')', "" );
        param = kernel.name.substr( pp + 1, kernel.name.size() - pp - 2 );
        bname = kernel.name.substr( 0, pp );
        gen_code( tmp_dir.p / "generated.h", bname, param );
    }

    // prerequisites for the types
    std::set<std::string> include_set, src_head_set, seen_types;
    std::ostringstream includes, src_heads;
    for( const std::string &input_type : input_types )
        get_prereq_req( includes, src_heads, include_set, src_head_set, seen_types, input_type );

    // header(s) and typedefs
    fcpp << includes.str();

    fcpp << "\n";
    fcpp << "#include <parex/TaskRef.h>\n";
    if ( generated )
        fcpp << "#include \"generated.h\"\n";
    else
        fcpp << "#include <" << bname << ".h>\n";

    fcpp << "\n";
    fcpp << src_heads.str();

    //
    fcpp << "\n";
    fcpp << "namespace {\n";
    fcpp << "    struct KernelWrapper {\n";
    fcpp << "        auto operator()( parex::Task *" << ( kernel.task_as_arg ? "task" : "/*task*/" ) << ", void **data ) const {\n";
    fcpp << "            return " << bname << "(\n";
    if ( kernel.task_as_arg )
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

void KernelCode::make_gen_cmk( TmpDir &tmp_dir ) {
    std::ofstream fcmk( tmp_dir.p / "CMakeLists.txt" );

    fcmk << "cmake_minimum_required(VERSION 3.0)\n";
    fcmk << "project(generator)\n";

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
}

void KernelCode::make_gen_cpp( TmpDir &tmp_dir, const path &output_path, const std::string &bname, const std::string &param ) {
    std::ofstream fcpp( tmp_dir.p / "generator.cpp" );

    // header(s) and typedefs
    fcpp << "#include <" << bname << ".h>\n";
    fcpp << "#include <fstream>\n";

    fcpp << "\n";
    fcpp << "int main( int, char **argv ) {\n";
    fcpp << "    std::ofstream fout( \"" << output_path.string() << "\" );\n";
    fcpp << "    " << bname << "( fout, \"" << bname << "\", \"" << cstr_encode( param ) << "\" );\n";
    fcpp << "}\n";
}

void KernelCode::gen_code( const path &output_path, const std::string &bname, const std::string &param ) {
    TmpDir tmp_dir;
    make_gen_cmk( tmp_dir );
    make_gen_cpp( tmp_dir, output_path, bname, param );

    build( tmp_dir.p, {} );

    exec( tmp_dir.p / "build" / "generator" );
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

void KernelCode::exec( const std::string &cmd ) {
    std::cout << cmd << std::endl;
    if ( system( cmd.c_str() ) ) {
        ERROR( "Error in cmd: {}", cmd );
    }
}

void KernelCode::init_base_types() {
    src_heads[ "ostream" ].push_back( "using std::ostream;" );

    src_heads[ "SI32"    ].push_back( "using SI32 = std::int32_t;" );
    includes [ "SI32"    ].push_back( "<cstdint>" );

    src_heads[ "SI64"    ].push_back( "using SI64 = std::int64_t;" );
    includes [ "SI64"    ].push_back( "<cstdint>" );

    src_heads[ "PI32"    ].push_back( "using PI32 = std::uint32_t;" );
    includes [ "PI32"    ].push_back( "<cstdint>" );

    src_heads[ "PI64"    ].push_back( "using PI64 = std::uint64_t;" );
    includes [ "PI64"    ].push_back( "<cstdint>" );

    src_heads[ "FP64"    ].push_back( "using FP64 = double;" );
    src_heads[ "FP32"    ].push_back( "using FP32 = float;" );

    includes [ "Tensor"  ].push_back( "<parex/containers/Tensor.h>" );
    src_heads[ "Tensor"  ].push_back( "using parex::Tensor;" );

    includes [ "Vec"     ].push_back( "<parex/containers/Vec.h>" );
    src_heads[ "Vec"     ].push_back( "using parex::Vec;" );

    includes [ "S"       ].push_back( "<parex/support/S.h>" );
    src_heads[ "S"       ].push_back( "using parex::S;" );

    includes [ "N"       ].push_back( "<parex/support/N.h>" );
    src_heads[ "N"       ].push_back( "using parex::N;" );
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

void KernelCode::build( const path &src_dir, const std::string &build_opt ) {
    path bld_dir = src_dir / "build";
    exec( "cmake -S '" + src_dir.string() + "' -B '" + bld_dir.string() + "' -DCMAKE_INSTALL_PREFIX='" + object_dir.string() + "'" ); //  > /dev/null
    exec( "cmake --build '" + bld_dir.string() + "'" + build_opt ); // > /dev/null
}

} // namespace parex
