#pragma once

#include "../plugin_managers/SrcSet.h"
#include "ComputableTask.h"

/**
*/
class CompiledTask : public ComputableTask {
public:
    /***/                  CompiledTask      ( std::vector<Rc<Task>> &&children, double priority = 0 );

    virtual void           exec              () override; ///< look in the cache, compile, load and run

    // new methods that must or can be surdefined
    virtual std::string    exported_func_name(); ///< name of the symbol in the library
    virtual std::string    called_func_name  (); ///< name of the function called from the exported one
    virtual void           get_src_content   ( Src &src, SrcSet &sw ) = 0;
    virtual std::string    summary           (); ///< by default, do not give aything, to let GeneratedLibrarySet make its own summary from the sources
};

