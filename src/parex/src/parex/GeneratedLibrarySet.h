#pragma once

#include <dynalo/dynalo.hpp>
#include "SrcWriter.h"
#include <filesystem>
#include <functional>

class SrcWriter;

/**
*/
class GeneratedLib {
public:
    using    Path            = std::filesystem::path;

    /**/     GeneratedLib    ( const std::string &name, const std::string &summary_of_parameters, const std::function<void( SrcWriter &sw )> &src_writer, const Path &base_output_directory = ".objects" );
    /**/     GeneratedLib    ();

    void*    symbol          ( const std::string &symbol_name );
    void     init            ( const std::string &name, const std::string &summary_of_parameters, const std::function<void( SrcWriter &sw )> &src_writer, const Path &base_output_directory = ".objects" );
    operator bool            () const;

private:
    using    LibPtr          = std::unique_ptr<dynalo::library>;

    void     compile_lib     ( const Path &output_directory, const std::string &shash, const std::function<void( SrcWriter &sw )> &src_writer );
    void     make_cmake      ( const Path &path, SrcWriter &sw, const std::string &shash );
    void     load_lib        ( const Path &lib_path );
    int      exec            ( std::string cmd ) const;

    LibPtr   lib;
};

