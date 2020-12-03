#pragma once

#include <filesystem>

/**
*/
class TmpDir {
public:
    using    path       = std::filesystem::path;

    /**/     TmpDir     ( std::string basename = "parex_" );
    /**/    ~TmpDir     ();
    operator std::string() const;

    path     p;
};
