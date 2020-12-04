#pragma once

#include "../plugin_managers/CompilationEnvironment.h"
#include "Type.h"

/**
*/
class CppType : public Type {
public:
    using                               VoidPtrFunc       = void(void *);

    /**/                                CppType           ( std::string name, const CompilationEnvironment &compilation_environment, std::vector<Type *> &&sub_types = {} );

    virtual UPType                      copy_with_sub_type( std::string name, std::vector<Type *> &&sub_types ) const override;
    virtual void                        for_each_type_rec ( const std::function<void(const Type *)> &cb ) const override;
    virtual void                        write_to_stream   ( std::ostream &os ) const override;
    virtual std::string                 cpp_name          () const override;
    virtual void                        destroy           ( void *data ) const override;

    mutable VoidPtrFunc*                destructor_func;  ///<
    std::vector<Type *>                 sub_types;        ///<
    std::string                         name;             ///<
};

