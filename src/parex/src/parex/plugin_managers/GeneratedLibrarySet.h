#pragma once

#include "DynamicLibrary.h"
#include <unordered_map>
#include <functional>
#include "SrcSet.h"

/**
*/
class GeneratedLibrarySet {
public:
    using           Path               = DynamicLibrary::Path;

    /**/            GeneratedLibrarySet( const Path &output_directory = ".generated_libs" );

    DynamicLibrary* get_library        ( const std::function<void(SrcSet&)> &src_writer, const std::string &summary_of_parameters = {}, SrcSet &&src_set = {} );

private:
    using           LibMap             = std::unordered_map<std::string,DynamicLibrary>;

    DynamicLibrary  load_or_make       ( const std::function<void(SrcSet&)> &src_writer, const std::string &summary_of_parameters, SrcSet &&src_set );
    void            compile_lib        ( const std::string &shash, const SrcSet &src_set );
    void            make_cmake         ( const Path &src_path, const std::string &shash, const SrcSet &src_set );
    int             exec_cmd           ( std::string cmd ) const;

    Path            output_directory;  ///< where lixxx.so and libxxx.so.info are written
    LibMap          lib_map;
};

