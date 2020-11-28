#include "GeneratedLibrarySet.h"
#include "TmpDir.h"
#include "ERROR.h"

#include <iostream>
#include <fstream>

GeneratedLibrarySet::GeneratedLibrarySet( const Path &output_directory ) : output_directory( output_directory ) {
}

DynamicLibrary *GeneratedLibrarySet::get_library( const std::function<void(SrcSet &)> &src_writer, const std::string &summary_of_parameters, SrcSet &&src_set ) {
    // if need to make a summary, use the sources
    if ( summary_of_parameters.empty() ) {
        if ( ! src_set )
            src_writer( src_set );
        return get_library( {}, src_set.summary(), std::move( src_set ) );
    }

    // test if already loaded
    auto iter = lib_map.find( summary_of_parameters );
    if ( iter == lib_map.end() )
        iter = lib_map.insert( iter, { summary_of_parameters, load_or_make( src_writer, summary_of_parameters, std::move( src_set ) ) } );
    return &iter->second;
}

DynamicLibrary GeneratedLibrarySet::load_or_make( const std::function<void(SrcSet &)> &src_writer, const std::string &summary_of_parameters, SrcSet &&src_set ) {
    // make a hash of the summary as a base to look in the filesystem
    std::hash<std::string> hasher;
    std::size_t hash = hasher( summary_of_parameters );

    // look if stored in the filesystem (looking for a .info file) ?
    for( ; ; ++hash ) {
        std::string shash = std::to_string( hash );
        Path plib = output_directory / dynalo::to_native_name( shash );
        Path pinfo = plib; pinfo.replace_extension( ".info" );

        // no file with corresponding name => create a new lib/info pair
        if ( ! std::filesystem::exists( pinfo ) || ! std::filesystem::exists( plib ) ) {
            if ( ! src_set )
                src_writer( src_set );
            compile_lib( shash, src_set );

            std::ofstream finfo( pinfo );
            finfo << summary_of_parameters;

            return { plib };
        }

        // else, if info is good, return the lib
        std::ifstream finfo( pinfo );
        std::ostringstream sinfo;
        sinfo << finfo.rdbuf();
        if ( sinfo.str() == summary_of_parameters )
            return { plib };
    }
}

void GeneratedLibrarySet::compile_lib( const std::string &shash, const SrcSet &src_set ) {
    TmpDir tmp_dir;

    // get the code
    src_set.write_files( tmp_dir.p );

    // make a CMakeLists.txt
    make_cmake( tmp_dir.p, shash, src_set );

    // compile
    Path bld_dir = tmp_dir.p / "build";
    exec_cmd( "cmake -S '" + tmp_dir.p.string() + "' -B '" + bld_dir.string() + "' -DCMAKE_INSTALL_PREFIX='" + output_directory.string() + "'" );
    exec_cmd( "cmake --build '" + bld_dir.string() + "' --target install --verbose" );
}

void GeneratedLibrarySet::make_cmake( const Path &src_path, const std::string &shash, const SrcSet &src_set ) {
    std::ofstream os( src_path / "CMakeLists.txt" );

    os << "cmake_minimum_required(VERSION 3.0)\n";
    os << "project(" << shash << ")\n";

    os << "\n";
    os << "add_library(" << shash << " SHARED\n";
    for( const auto &p : src_set.src_map )
        os << "    " << p.first.string() << "\n";
    os << ")";

    os << "\n";
    for( const auto &p : src_set.src_map ) {
        if ( ! p.second.cpp_flags.empty() ) {
            os << "\nset_property(SOURCE " << p.first.string() << " PROPERTY COMPILE_OPTIONS";
            for( const auto &cpp_flag : p.second.cpp_flags )
                os << " " << cpp_flag;
            os << ")";
        }

        if ( ! p.second.include_directories.empty() ) {
            os << "\nset_property(SOURCE " << p.first.string() << " PROPERTY INCLUDE_DIRECTORIES";
            for( const auto &include_directory : p.second.include_directories )
                os << "\n    " << std::filesystem::absolute( include_directory ).string();
            os << "\n)";
        }
    }

    os << "\n";
    os << "install(TARGETS " << shash << " DESTINATION .)\n";
}

int GeneratedLibrarySet::exec_cmd( std::string cmd ) const {
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

