#pragma once

#include "../plugins/CompilationEnvironment.h"
#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace parex {
namespace hardware_information { class Memory; }
class HwGraph;
class SrcSet;
class Src;

/**
*/
class Type {
public:
    using                  MemoriesFunc             = void(VecUnique<hardware_information::Memory> &memories, const HwGraph *hw_graph, const void *);
    using                  DestroyFunc              = void(void *);
    using                  Memory                   = hardware_information::Memory;

    /**/                   Type                     ( const std::string &name = {}, CompilationEnvironment &&compilation_environment = {}, const std::string &base_name = {}, std::vector<Type *> &&sub_types = {} );
    virtual               ~Type                     ();

    virtual void           write_to_stream          ( std::ostream &os ) const;
    virtual void           memories                 ( VecUnique<Memory> &memories, const void *data ) const;
    virtual void           destroy                  ( void *data ) const;

    virtual void           gen_func_memories        ( Src &src, SrcSet &sw ) const;
    virtual void           gen_func_destroy         ( Src &src, SrcSet &sw ) const;

    CompilationEnvironment compilation_environment; ///<
    mutable MemoriesFunc*  memories_func;           ///<
    mutable DestroyFunc*   destroy_func;            ///<
    std::string            base_name;               ///<
    std::vector<Type *>    sub_types;               ///<
    std::string            name;                    ///<
};

} // namespace parex
