#pragma once

#include <vector>
#include <string>

/**
*/
class TypeFactoryRegistrar {
public:
    /**/                     TypeFactoryRegistrar( std::string name, std::vector<std::string> includes = {}, std::vector<std::string> preliminaries = {}, std::vector<std::string> include_directories = {} );

    std::string              name;
    std::vector<std::string> includes;
    std::vector<std::string> preliminaries;
    std::vector<std::string> include_directories;

    TypeFactoryRegistrar    *prev_type_factory_registrar;
};

extern TypeFactoryRegistrar *last_type_factory_registrar;
