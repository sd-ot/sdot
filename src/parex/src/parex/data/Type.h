#pragma once

#include "../plugins/CompilationEnvironment.h"
#include <functional>
#include <ostream>
#include <memory>
#include <string>
#include <vector>

namespace parex {
class Src;

/**
*/
class Type {
public:
    using                  UPType                   = std::unique_ptr<Type>;

    virtual               ~Type                     ();

    virtual UPType         copy_with_sub_type       ( std::string name, std::vector<Type *> &&sub_types ) const = 0;
    virtual void           for_each_type_rec        ( const std::function<void(const Type *)> &cb ) const;
    virtual void           write_to_stream          ( std::ostream &os ) const = 0;
    virtual std::string    cpp_name                 () const = 0;
    virtual void           destroy                  ( void *data ) const = 0;

    void                   add_needs_in             ( Src &src ) const;

    CompilationEnvironment compilation_environment; ///<
};

} // namespace parex
