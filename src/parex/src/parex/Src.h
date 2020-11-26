#pragma once

#include "VecUnique.h"
#include <filesystem>
#include <sstream>
#include <string>


/**
*/
class Src {
public:
    using                  String               = std::string;
    using                  Path                 = std::filesystem::path;
    using                  VUS                  = VecUnique<String>;

    /**/                   Src                  ( VUS include_directories, VUS cpp_flags, VUS includes, VUS prelims );

    template<class T> Src& operator<<           ( const T &value ) { fout << value; return *this; }
    void                   write_to             ( std::ostream &os ) const;

    VUS                    include_directories; ///<
    VUS                    cpp_flags;           ///<
    VUS                    includes;            ///<
    VUS                    prelims;             ///<
    std::ostringstream     fout;                ///<
};

