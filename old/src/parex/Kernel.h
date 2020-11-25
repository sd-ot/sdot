#pragma once

#include <limits>
#include <string>
#include <tuple>

namespace parex {

/**
*/
class Kernel {
public:
    bool          operator<             ( const Kernel &that ) const;
    operator      bool                  () const { return name.size(); }

    /**/          Kernel                ( std::string name, int priority = 0, bool task_as_arg = false ) : name( name ), priority( priority ), task_as_arg( task_as_arg ), vararg_num( std::numeric_limits<unsigned>::max() ) {}
    /**/          Kernel                ( const char *name, int priority = 0, bool task_as_arg = false ) : Kernel( std::string( name ), priority, task_as_arg ) {}
    /**/          Kernel                () : Kernel( std::string{} ) {}

    static Kernel with_task_as_arg      ( const Kernel &that ) { Kernel res = that; res.task_as_arg = true; return res; }
    static Kernel with_vararg_num       ( unsigned vararg_num, std::string vararg_default_type, std::string vararg_enforced_type, const Kernel &that );
    static Kernel with_priority         ( int priority, const Kernel &that ) { Kernel res = that; res.priority = priority; return res; }

    bool          has_varargs           () const { return vararg_num < std::numeric_limits<unsigned>::max(); }
    auto          tie                   () const { return std::tie( name, priority, vararg_num, vararg_default_type, vararg_enforced_type, task_as_arg ); }

    std::string   name;                 ///<
    int           priority;             ///<
    bool          task_as_arg;          ///<

    unsigned      vararg_num;           ///<
    std::string   vararg_default_type;  ///<
    std::string   vararg_enforced_type; ///<
};

} // namespace parex
