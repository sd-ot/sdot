#include "GeneratedLib.h"
#include "TmpDir.h"
#include "ERROR.h"
#include "P.h"

#include <iostream>
#include <fstream>

GeneratedLib::GeneratedLib( const std::string &name, const std::string &summary_of_parameters, const std::function<void( SrcWriter &sw )> &src_writer, const Path &base_output_directory ) {
    init( name, summary_of_parameters, src_writer, base_output_directory );
}

GeneratedLib::GeneratedLib() {
}

void GeneratedLib::init( const std::string &name, const std::string &summary_of_parameters, const std::function<void (SrcWriter &)> &src_writer, const GeneratedLib::Path &base_output_directory ) {
    Path output_directory = base_output_directory / name;

    std::hash<std::string> hasher;
    std::size_t hash = hasher( summary_of_parameters );

    // look if stored in the filesystem (looking for a .info file) ?
    for( ; ; ++hash ) {
        std::string shash = std::to_string( hash );
        Path pinfo = output_directory / ( shash + ".info" );
        Path plib = output_directory / dynalo::to_native_name( shash );

        // no file with corresponding name => create a new lib/info pair
        if ( ! std::filesystem::exists( pinfo ) || ! std::filesystem::exists( plib ) ) {
            compile_lib( output_directory, shash, src_writer );
            load_lib( plib );

            std::ofstream finfo( pinfo );
            finfo << summary_of_parameters;
            return;
        }

        // else, if info is good, return the lib
        std::ifstream finfo( pinfo );
        std::ostringstream sinfo;
        sinfo << finfo.rdbuf();
        if ( sinfo.str() == summary_of_parameters ) {
            load_lib( plib );
            return;
        }
    }
}

GeneratedLib::operator bool() const {
    return lib.get();
}

void *GeneratedLib::symbol( const std::string &symbol_name ) {
    return lib->get_function<void *>( symbol_name );
}

void GeneratedLib::compile_lib( const Path &output_directory, const std::string &shash, const std::function<void( SrcWriter &sw )> &src_writer ) {
    TmpDir tmp_dir;

    // get the code
    SrcWriter sw;
    src_writer( sw );
    for( const auto &p : sw.src_map ) {
        std::ofstream fout( tmp_dir.p / p.first );
        p.second.write_to( fout );
    }

    // make a CMakeLists.txt
    make_cmake( tmp_dir.p, sw, shash );

    // compile
    Path bld_dir = tmp_dir.p / "build";
    exec( "cmake -S '" + tmp_dir.p.string() + "' -B '" + bld_dir.string() + "' -DCMAKE_INSTALL_PREFIX='" + output_directory.string() + "'" );
    exec( "cmake --build '" + bld_dir.string() + "' --target install --verbose" );
}

void GeneratedLib::load_lib( const Path &lib_path ) {
    lib = std::make_unique<dynalo::library>( lib_path.string() );
}

void GeneratedLib::make_cmake( const Path &path, SrcWriter &sw, const std::string &shash ) {
    std::ofstream os( path / "CMakeLists.txt" );

    os << "cmake_minimum_required(VERSION 3.0)\n";
    os << "project(" << shash << ")\n";

    os << "\n";
    os << "add_library(" << shash << " SHARED\n";
    for( const auto &p : sw.src_map )
        os << "    " << p.first << "\n";
    os << ")";

    os << "\n";
    for( const auto &p : sw.src_map ) {
        if ( ! p.second.cpp_flags.empty() ) {
            os << "\nset_property(SOURCE " << p.first << " PROPERTY COMPILE_OPTIONS";
            for( const auto &cpp_flag : p.second.cpp_flags )
                os << " " << cpp_flag;
            os << ")";
        }

        if ( ! p.second.include_directories.empty() ) {
            os << "\nset_property(SOURCE " << p.first << " PROPERTY INCLUDE_DIRECTORIES";
            for( const auto &include_directory : p.second.include_directories )
                os << "\n    " << std::filesystem::absolute( include_directory ).string();
            os << "\n)";
        }
    }

    os << "\n";
    os << "install(TARGETS " << shash << " DESTINATION .)\n";
}

int GeneratedLib::exec( std::string cmd ) const {
    // std::ostream &fout = std::cout; //std::ofstream fout( log, std::ios_base::app );
    // fout << "=======================\n";
    // fout << cmd << "\n";

    // cmd += " 2>&1 > /dev/null"; // + log.string();
    cmd += " > /dev/null"; // + log.string();
    int res = system( cmd.c_str() ); // cmd += " 2>&1 > " + log.string();
    if ( res )
        ERROR( "Error in cmd: {}", cmd ); // ERROR( "Error in cmd: {}\nSee log file '{}'", cmd, log.string() );

    return res;
}

