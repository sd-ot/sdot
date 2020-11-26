#pragma once

#include <filesystem>

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
