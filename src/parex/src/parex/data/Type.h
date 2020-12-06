#pragma once

#include "../plugins/CompilationEnvironment.h"
#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace parex {
namespace hardware_information { class Memory; }
class Src;

/**
*/
class Type {
public:
    using                  VoidPtrFunc              = void(void *);
    using                  Memory                   = hardware_information::Memory;

    virtual               ~Type                     ();

    virtual void           write_to_stream          ( std::ostream &os ) const;
    virtual void           destroy                  ( void *data ) const;

    CompilationEnvironment compilation_environment; ///<
    mutable VoidPtrFunc*   destructor_func;         ///<
    VecUnique<Memory *>    used_memories;           ///<
    std::string            base_name;               ///<
    std::vector<Type *>    sub_types;               ///<
    std::string            name;                    ///<
};

} // namespace parex
