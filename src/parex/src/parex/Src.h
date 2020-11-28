#pragma once

#include "VecUnique.h"
#include <filesystem>
#include <sstream>
#include <string>


/**
*/
class Src {
public:
    using                  Path                 = std::filesystem::path;
    using                  VUPath               = VecUnique<Path>;
    using                  VUString             = VecUnique<std::string>;

    /**/                   Src                  ( VUPath include_directories, VUString cpp_flags, VUString includes, VUString prelims );

    template<class T> Src& operator<<           ( const T &value ) { fout << value; return *this; }
    void                   write_to             ( std::ostream &os ) const;

    VUPath                 include_directories; ///<
    VUString               cpp_flags;           ///<
    VUString               includes;            ///<
    VUString               prelims;             ///<
    std::ostringstream     fout;                ///<
};

