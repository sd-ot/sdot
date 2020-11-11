#pragma once

#include <string>

namespace parex {

/**
*/
class Kernel {
public:
    bool          operator<       ( const Kernel &that ) const;
    operator      bool            () const { return name.size(); }

    /**/          Kernel          ( std::string name, int priority = 0, bool task_as_arg = false ) : name( name ), priority( priority ), task_as_arg( task_as_arg ) {}
    /**/          Kernel          ( const char *name, int priority = 0, bool task_as_arg = false ) : name( name ), priority( priority ), task_as_arg( task_as_arg ) {}
    /**/          Kernel          () : priority( 0 ) {}

    static Kernel with_task_as_arg( const Kernel &that ) { Kernel res = that; res.task_as_arg = true; return res; }
    static Kernel with_priority   ( int priority, const Kernel &that ) { Kernel res = that; res.priority = priority; return res; }

    std::string   name;           ///<
    int           priority;       ///<
    bool          task_as_arg;    ///<
};

} // namespace parex
