#pragma once

#include "DynamicSymbol.h"
#include "VecUnique.h"
#include "Type.h"

/**
*/
class CppType : public Type {
public:
    using                               VoidPtrFunc               = void(void *);

    /**/                                CppType                   ( std::string name, VecUnique<std::string> include_directories = {}, VecUnique<std::string> includes = {}, VecUnique<std::string> preliminaries = {}, std::vector<Type *> &&sub_types = {} );

    virtual void                        for_each_include_directory( const std::function<void(std::string)> &cb ) const override;
    virtual UPType                      copy_with_sub_type        ( std::string name, std::vector<Type *> &&sub_types ) const override;
    virtual void                        for_each_include          ( const std::function<void(std::string)> &cb ) const override;
    virtual void                        for_each_prelim           ( const std::function<void(std::string)> &cb ) const override;
    virtual void                        write_to_stream           ( std::ostream &os ) const override;
    virtual std::string                 cpp_name                  () const override;
    virtual void                        destroy                   ( void *data ) const override;

    VecUnique<std::string>              include_directories;      ///<
    mutable DynamicSymbol<void(void *)> destructor_func;          ///<
    VecUnique<std::string>              preliminaries;            ///<
    std::vector<Type *>                 sub_types;                ///<
    VecUnique<std::string>              includes;                 ///<
    std::string                         name;                     ///<
};

