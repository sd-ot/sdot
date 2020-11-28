#include "CompiledSymbolMap.h"
#include "TmpDir.h"
#include "ERROR.h"
#include "TODO.h"
#include "P.h"

void *CompiledSymbolMap::untyped_symbol_for( const std::string &parameters ) {
    // look in already loaded functions
    auto iter = df_map.find( parameters );
    if ( iter == df_map.end() )
        iter = df_map.insert( iter, { parameters, load_or_make_lib( parameters ) } );
    return iter->second.sym;
}

CompiledSymbolMap::DF CompiledSymbolMap::load_or_make_lib( const std::string &parameters ) {
    std::hash<std::string> hasher;
    Path op = output_directory( parameters );
    std::string symbol_name = this->symbol_name( parameters );
    for( std::size_t hash = hasher( parameters ); ; ++hash ) {
        std::string shash = std::to_string( hash );
        std::string ninfo = shash + ".info";
        Path pinfo = op / ninfo;

        // no file with corresponding name => create a new lib/info pair
        if ( ! std::filesystem::exists( pinfo ) )
            return make_lib( op, pinfo, shash, parameters, symbol_name );

        // else, if info is good, return the lib
        std::ifstream finfo( pinfo );
        std::ostringstream sinfo;
        sinfo << finfo.rdbuf();
        if ( sinfo.str() == parameters )
            return load_sym( op, shash, symbol_name );
    }
}

CompiledSymbolMap::DF CompiledSymbolMap::load_sym( const Path &output_directory, const std::string &shash, const std::string &symbol_name ) {
    Path p = output_directory / "lib" / dynalo::to_native_name( shash );

    DF res;
    res.lib = std::make_unique<dynalo::library>( p.string() );
    res.sym = res.lib->get_function<void *>( symbol_name );
    if ( ! res.sym )
        ERROR( "Impossible to find symbol '{}' in '{}'", symbol_name, p.string() );
    return res;
}

CompiledSymbolMap::DF CompiledSymbolMap::make_lib( const Path &output_directory, const Path &pinfo, const std::string &shash, const std::string &parameters, const std::string &symbol_name ) {
    TmpDir tmp_dir;

    // make source files
    SrcSet ff;
    ff.default_include_directories = { PAREX_DIR "/src" };
    ff.default_cpp_flags = "-g3 -march=native -O3";
    ff.compiled_symbol_map = this;
    ff.symbol_name = symbol_name;
    ff.parameters = parameters;
    ff.tmp_dir = &tmp_dir;
    make_srcs( ff );
    ff.close();

    // make CMakeLists.txt
    Path cmk_path = tmp_dir.p / "CMakeLists.txt";
    std::ofstream fcmk( cmk_path );
    make_cmakelists( fcmk, ff, shash );
    fcmk.close();

    // build .so file
    Path bld_dir = tmp_dir.p / "build";
    exec( "cmake -S '" + tmp_dir.p.string() + "' -B '" + bld_dir.string() + "' -DCMAKE_INSTALL_PREFIX='" + output_directory.string() + "'" );
    exec( "cmake --build '" + bld_dir.string() + "' --target install --verbose" );

    // make .info file
    std::ofstream fkstr( pinfo );
    fkstr << parameters;

    // load lib
    return load_sym( output_directory, shash, symbol_name );
}

int CompiledSymbolMap::exec( const std::string &cmd ) const {
    std::ostream &fout = std::cout; //std::ofstream fout( log, std::ios_base::app );
    fout << "=======================\n";
    fout << cmd << "\n";

    int res = system( cmd.c_str() ); // cmd += " 2>&1 > " + log.string();
    if ( res )
        ERROR( "Error in cmd: {}", cmd ); // ERROR( "Error in cmd: {}\nSee log file '{}'", cmd, log.string() );

    return res;
}

std::string CompiledSymbolMap::symbol_name( const std::string &/*parameters*/ ) const {
    return "symbol_to_load";
}

void CompiledSymbolMap::make_cmakelists( std::ostream &os, const SrcSet &ff, const std::string &shash ) const {
    os << "cmake_minimum_required(VERSION 3.0)\n";
    os << "project(" << shash << ")\n";

    os << "\n";
    os << "add_library(" << shash << " SHARED\n";
    for( const auto &p : ff.src_map )
        os << "    " << p.first << "\n";
    os << ")";

    os << "\n";
    for( const auto &p : ff.src_map ) {
        if ( ! p.second.cpp_flags.empty() )
            os << "\nset_property(SOURCE " << p.first << " PROPERTY COMPILE_OPTIONS " << p.second.cpp_flags << ")";

        if ( ! p.second.include_directories.empty() ) {
            os << "\nset_property(SOURCE " << p.first << " PROPERTY INCLUDE_DIRECTORIES";
            for( const auto &include_directory : p.second.include_directories )
                os << " " << include_directory;
            os << ")";
        }
    }

    //    os << "\n";
    //    os << "add_definitions(-DPAREX_IN_KERNEL)\n";

    os << "\n";
    os << "install(TARGETS " << shash << ")\n";
}
