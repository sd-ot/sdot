#pragma once

#include <filesystem>

namespace parex {

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

} // namespace parex
