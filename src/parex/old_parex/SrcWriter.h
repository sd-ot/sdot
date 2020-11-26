#pragma once

#include "Src.h"
#include <map>

class CompiledSymbolMap;
class TmpDir;

/**
*/
class SrcWriter {
public:
    using                    SrcMap                       = std::map<std::string,Src>;
    using                    Path                         = Src::Path;

    template<class T> Src   &operator<<                   ( const T &value ) { return src( main_cpp_name() ) << value; }
    void                     close                        ();
    Src&                     src                          ( std::string filename );

    CompiledSymbolMap       *compiled_symbol_map;         ///<
    TmpDir                  *tmp_dir;                     ///<

    std::vector<std::string> default_include_directories; ///<
    std::string              default_cpp_flags;           ///<
    std::vector<std::string> default_includes;            ///<
    std::string              symbol_name;                 ///<
    std::string              parameters;                  ///<

private:
    friend class             CompiledSymbolMap;           ///<
    std::string              main_cpp_name                () const;

    SrcMap                   src_map;                     ///<
};

