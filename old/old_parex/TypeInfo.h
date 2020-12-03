#pragma once

#include <functional>
#include <string>

/**
*/
class TypeInfo {
public:
    virtual            ~TypeInfo         ();

    virtual void        get_preliminaries( const std::function<void(const std::string &)> &f );
    virtual void        get_includes     ( const std::function<void(const std::string &)> &f );
    virtual std::string cpp_name         () = 0;
};

