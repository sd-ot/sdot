#pragma once

#include <parex/ComputableTask.h>
#include <parex/Value.h>

namespace sdot {

/**
*/
class GetElementaryPolytopInfoListContent : public ComputableTask {
public:
    /***/               GetElementaryPolytopInfoListContent( const Rc<Task> &shape_types );
    virtual void        write_to_stream                    ( std::ostream &os ) const override;
    virtual void        exec                               () override;

private:
    static std::string  default_shape_types                ( int dim );
    static void         write_ctor                         ( std::ostream &os, std::istringstream &&shape_types, const std::string &sp );
};

} // namespace sdot
