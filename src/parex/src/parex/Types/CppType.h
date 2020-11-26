#pragma once

#include "../GeneratedSym.h"
#include "../Type.h"
#include <vector>

/**
*/
class CppType : public Type {
public:
    using                              VoidPtrFunc     = void(void *);

    /**/                               CppType         ( std::string name, std::vector<std::string> includes = {}, std::vector<std::string> preliminaries = {} );

    virtual void                       for_each_include( const std::function<void(std::string)> &cb ) const override;
    virtual void                       for_each_prelim ( const std::function<void(std::string)> &cb ) const override;
    virtual void                       write_to_stream ( std::ostream &os ) const override;
    virtual std::string                cpp_name        () const override;
    virtual void                       destroy         ( void *data ) const override;

    mutable GeneratedSym<void(void *)> destructor_func;
    std::vector<std::string>           preliminaries;
    std::vector<std::string>           includes;
    std::string                        name;
};

