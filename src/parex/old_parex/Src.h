#pragma once

#include <filesystem>
#include <fstream>
#include <vector>
#include <string>


/**
*/
class Src {
public:
    using                    Path                 = std::filesystem::path;

    /**/                     Src                  ( const Path &filename, std::string default_cpp_flags, std::vector<std::string> default_includes, std::vector<std::string> default_include_directories );

    void                     add_include_directory( std::string include );
    void                     add_cpp_flags        ( std::string cpp_flags );
    void                     add_include          ( std::string include );
    template<class T> Src   &operator<<           ( const T &value ) { fout << value; return *this; }

    std::vector<std::string> include_directories; ///<
    std::string              cpp_flags;           ///<
    std::vector<std::string> includes;            ///<
    std::ofstream            fout;                ///<
};

