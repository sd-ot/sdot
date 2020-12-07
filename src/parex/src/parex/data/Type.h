#pragma once

#include "../plugins/CompilationEnvironment.h"
#include "../hardware/Memory.h"
#include "../utility/Rc.h"
#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace parex {
namespace hardware_information { class Memory; }
class HwGraph;
class SrcSet;
class Task;
class Src;

/**
*/
class Type {
public:
    using                    MemoriesFunc             = void(VecUnique<hardware_information::Memory *> &memories, const HwGraph *hw_graph, const void *);
    using                    DestroyFunc              = void(void *);
    using                    Memory                   = hardware_information::Memory;

    /**/                     Type                     ( CompilationEnvironment &&compilation_environment = {}, std::vector<Type *> &&sub_types = {} );
    virtual                 ~Type                     ();

    virtual void             write_to_stream          ( std::ostream &os ) const;
    virtual Rc<Task>         conv_alloc_task          ( const Rc<Task> &task, hardware_information::Memory *dst ) const; ///< new task to convert to a type allocated in dst
    virtual void             get_memories             ( VecUnique<hardware_information::Memory *> &memories, const void *data ) const;
    virtual void             destroy                  ( void *data ) const;

    virtual void             gen_func_get_memories    ( Src &src, SrcSet &sw ) const;
    virtual void             gen_func_destroy         ( Src &src, SrcSet &sw ) const;

    CompilationEnvironment   compilation_environment; ///<
    mutable MemoriesFunc*    get_memories_func;       ///<
    mutable DestroyFunc*     destroy_func;            ///<
    std::vector<std::string> parameters;              ///<
    std::string              base_name;               ///<
    std::vector<Type *>      sub_types;               ///<
    std::string              name;                    ///<
};

} // namespace parex
