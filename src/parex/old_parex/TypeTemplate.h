#pragma once

#include <vector>
#include <string>
/**
*/
class TypeTemplate {
public:
    /**/                TypeTemplate();
    virtual            ~TypeTemplate();

    virtual std::string cpp_name    ( const std::vector<std::string> &parameters ) const = 0;
};

