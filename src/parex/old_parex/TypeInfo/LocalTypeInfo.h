#pragma once

#include "../TypeInfo.h"
#include <vector>

/**
*/
class LocalTypeInfo : public TypeInfo {
public:
    /**/                     LocalTypeInfo    ( const std::string &name, const std::vector<std::string> &includes = {}, const std::string &preliminary = {} );

    virtual void             get_preliminaries( const std::function<void(const std::string &)> &f ) override;
    virtual void             get_includes     ( const std::function<void(const std::string &)> &f ) override;
    virtual std::string      cpp_name         () override;

    std::string              preliminary;     ///<
    std::vector<std::string> includes;        ///<
    std::string              name;            ///<
};

