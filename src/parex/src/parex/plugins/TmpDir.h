#pragma once

#include <filesystem>

namespace parex {

/**
*/
class TmpDir {
public:
    using    Path       = std::filesystem::path;

    /**/     TmpDir     ( std::string basename = "parex_" );
    /**/    ~TmpDir     ();
    operator std::string() const;

    Path     p;
};

} // namespace parex
