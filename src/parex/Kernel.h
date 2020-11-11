#pragma once

#include <string>

namespace parex {

/**
*/
class Kernel {
public:
    bool          operator<       ( const Kernel &that ) const;
    operator      bool            () const { return name.size(); }

    /**/          Kernel          ( std::string name, bool task_as_arg = false ) : name( name ), task_as_arg( task_as_arg ) {}
    /**/          Kernel          ( const char *name, bool task_as_arg = false ) : name( name ), task_as_arg( task_as_arg ) {}
    /**/          Kernel          () {}

    static Kernel with_task_as_arg( std::string name ) { return { name, true }; }

    std::string   name;           ///<
    bool          task_as_arg;    ///<
};

} // namespace parex
